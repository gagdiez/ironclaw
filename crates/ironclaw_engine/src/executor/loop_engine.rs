//! Core execution loop — the replacement for `run_agentic_loop()`.
//!
//! The `ExecutionLoop` owns a thread and drives it through LLM call →
//! action execution → result processing → repeat cycles. Unlike the
//! existing delegate pattern, the loop is self-contained: all behavior
//! differences between thread types are handled via capability leases
//! and policy, not delegate implementations.

use std::sync::Arc;

use tracing::debug;

use crate::capability::lease::LeaseManager;
use crate::capability::policy::PolicyEngine;
use crate::runtime::messaging::{SignalReceiver, ThreadOutcome};
use crate::traits::effect::EffectExecutor;
use crate::traits::llm::LlmBackend;
use crate::types::error::EngineError;
use crate::types::event::EventKind;
use crate::types::message::ThreadMessage;
use crate::types::step::Step;
use crate::types::thread::{Thread, ThreadState};

const RUNTIME_CHECKPOINT_METADATA_KEY: &str = "runtime_checkpoint";

/// Persisted state from a prior execution, used to resume threads.
/// The Python orchestrator manages loop counters internally; Rust only
/// needs the opaque `persisted_state` blob to hand back on resume.
#[derive(Default)]
struct RuntimeCheckpoint {
    persisted_state: serde_json::Value,
}

/// The core execution loop for a thread.
pub struct ExecutionLoop {
    pub thread: Thread,
    llm: Arc<dyn LlmBackend>,
    effects: Arc<dyn EffectExecutor>,
    leases: Arc<LeaseManager>,
    policy: Arc<PolicyEngine>,
    signal_rx: SignalReceiver,
    /// Stored for potential future use (e.g. user-scoped prompt overlays).
    _user_id: String,
    /// Optional capability registry for resolving capability-level policies.
    capabilities: Option<Arc<crate::capability::registry::CapabilityRegistry>>,
    /// Optional broadcast sender for live event streaming.
    event_tx: Option<tokio::sync::broadcast::Sender<crate::types::event::ThreadEvent>>,
    /// Optional retrieval engine for injecting prior knowledge into context.
    retrieval: Option<crate::memory::RetrievalEngine>,
    /// Optional Store for runtime prompt overlay loading and skill retrieval.
    store: Option<Arc<dyn crate::traits::store::Store>>,
    /// Runtime platform metadata for self-awareness in system prompts.
    platform_info: Option<crate::executor::prompt::PlatformInfo>,
}

impl ExecutionLoop {
    pub fn new(
        thread: Thread,
        llm: Arc<dyn LlmBackend>,
        effects: Arc<dyn EffectExecutor>,
        leases: Arc<LeaseManager>,
        policy: Arc<PolicyEngine>,
        signal_rx: SignalReceiver,
        user_id: String,
    ) -> Self {
        Self {
            thread,
            llm,
            effects,
            leases,
            policy,
            signal_rx,
            _user_id: user_id,
            capabilities: None,
            event_tx: None,
            retrieval: None,
            store: None,
            platform_info: None,
        }
    }

    /// Set the event broadcast sender for live status updates.
    pub fn with_event_tx(
        mut self,
        tx: tokio::sync::broadcast::Sender<crate::types::event::ThreadEvent>,
    ) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Set the capability registry for resolving capability-level policies.
    pub fn with_capabilities(
        mut self,
        capabilities: Arc<crate::capability::registry::CapabilityRegistry>,
    ) -> Self {
        self.capabilities = Some(capabilities);
        self
    }

    /// Set the retrieval engine for injecting prior knowledge into context.
    pub fn with_retrieval(mut self, retrieval: crate::memory::RetrievalEngine) -> Self {
        self.retrieval = Some(retrieval);
        self
    }

    /// Set the Store for runtime prompt overlay loading and skill retrieval.
    pub fn with_store(mut self, store: Arc<dyn crate::traits::store::Store>) -> Self {
        self.store = Some(store);
        self
    }

    /// Set platform metadata for self-awareness in system prompts.
    pub fn with_platform_info(mut self, info: crate::executor::prompt::PlatformInfo) -> Self {
        self.platform_info = Some(info);
        self
    }

    /// Add an event to the thread and broadcast it for live status updates.
    fn emit_event(&mut self, kind: EventKind) {
        let event = crate::types::event::ThreadEvent::new(self.thread.id, kind);
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event.clone());
        }
        self.thread.events.push(event);
        self.thread.updated_at = chrono::Utc::now();
    }

    fn load_runtime_checkpoint(&self) -> RuntimeCheckpoint {
        let persisted_state = self
            .thread
            .metadata
            .get(RUNTIME_CHECKPOINT_METADATA_KEY)
            .and_then(|value| value.get("persisted_state"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        RuntimeCheckpoint { persisted_state }
    }

    fn clear_runtime_checkpoint(&mut self) {
        if let Some(metadata) = self.thread.metadata.as_object_mut() {
            metadata.remove(RUNTIME_CHECKPOINT_METADATA_KEY);
        }
        self.thread.updated_at = chrono::Utc::now();
    }

    async fn persist_runtime_state(
        &self,
        step: Option<&Step>,
        persisted_event_count: &mut usize,
    ) -> Result<(), EngineError> {
        let Some(store) = self.store.as_ref() else {
            return Ok(());
        };

        // All three store writes are independent — run them in parallel.
        let step_fut = async {
            if let Some(step) = step {
                store.save_step(step).await
            } else {
                Ok(())
            }
        };

        let new_event_count = self.thread.events.len();
        let events_fut = async {
            if *persisted_event_count < new_event_count {
                store
                    .append_events(&self.thread.events[*persisted_event_count..])
                    .await
            } else {
                Ok(())
            }
        };

        let thread_fut = store.save_thread(&self.thread);

        let (step_res, events_res, thread_res) = tokio::join!(step_fut, events_fut, thread_fut);
        step_res?;
        events_res?;
        thread_res?;

        *persisted_event_count = new_event_count;
        Ok(())
    }

    /// Run the execution loop to completion.
    pub async fn run(&mut self) -> Result<ThreadOutcome, EngineError> {
        let mut persisted_event_count = self.thread.events.len();
        let checkpoint = self.load_runtime_checkpoint();

        // Transition to Running if this is a fresh start or restart from a resumable state.
        if self.thread.state != ThreadState::Running {
            self.thread.transition_to(ThreadState::Running, None)?;
        }

        // Pre-fetch shared memory docs once — used by both prompt overlay and
        // orchestrator loading, avoiding a duplicate Store query.
        let system_docs = if let Some(store) = self.store.as_ref() {
            match store.list_shared_memory_docs(self.thread.project_id).await {
                Ok(docs) => docs,
                Err(e) => {
                    debug!("failed to load shared docs for orchestrator: {e}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Inject CodeAct/RLM system prompt if none exists
        if !self
            .thread
            .messages
            .iter()
            .any(|m| m.role == crate::types::message::MessageRole::System)
        {
            // Fetch active leases (needed for action list)
            let active_leases = self.leases.active_for_thread(self.thread.id).await;
            let actions = match self.effects.available_actions(&active_leases).await {
                Ok(a) => a,
                Err(e) => {
                    debug!(thread_id = %self.thread.id, "failed to load actions for system prompt: {e}");
                    Vec::new()
                }
            };
            // Build prompt using pre-fetched docs (no extra Store query)
            let system_prompt = crate::executor::prompt::build_codeact_system_prompt_with_docs(
                &actions,
                &system_docs,
                self.platform_info.as_ref(),
            );

            // Skill selection and injection happens in the Python orchestrator
            // via __list_skills__() host function — not here in Rust.

            self.thread
                .messages
                .insert(0, ThreadMessage::system(system_prompt));
        }
        self.persist_runtime_state(None, &mut persisted_event_count)
            .await?;

        // Load versioned Python orchestrator using pre-fetched docs.
        // Self-modification is disabled by default — only the compiled-in v0
        // runs unless explicitly opted in via ORCHESTRATOR_SELF_MODIFY=true.
        // The flag is read from the process-wide snapshot (set once on first
        // call) so a runtime env mutation cannot flip the gate mid-task.
        let allow_self_modify = crate::runtime::self_modify_enabled();
        let (orchestrator_code, orchestrator_version) =
            crate::executor::orchestrator::load_orchestrator_from_docs(
                &system_docs,
                allow_self_modify,
            );

        debug!(
            thread_id = %self.thread.id,
            orchestrator_version,
            "running Python orchestrator"
        );

        // Store version in thread metadata for rollback tracking
        if let Some(metadata) = self.thread.metadata.as_object_mut() {
            metadata.insert(
                "orchestrator_version".into(),
                serde_json::json!(orchestrator_version),
            );
        }

        // Execute the Python orchestrator with host function dispatch
        let result = crate::executor::orchestrator::execute_orchestrator(
            &orchestrator_code,
            &mut self.thread,
            &self.llm,
            &self.effects,
            &self.leases,
            &self.policy,
            &mut self.signal_rx,
            self.event_tx.as_ref(),
            self.retrieval.as_ref(),
            self.store.as_ref(),
            &checkpoint.persisted_state,
        )
        .await;

        // Post-cleanup: persist final state, track failures for auto-rollback
        match result {
            Ok(orch_result) => {
                // Reset failure counter on success
                if let Some(store) = self.store.as_ref() {
                    crate::executor::orchestrator::reset_orchestrator_failures(
                        store,
                        self.thread.project_id,
                    )
                    .await;
                }
                let _ = &orch_result.tokens_used;

                self.clear_runtime_checkpoint();
                self.persist_runtime_state(None, &mut persisted_event_count)
                    .await?;
                Ok(orch_result.outcome)
            }
            Err(e) => {
                debug!(
                    thread_id = %self.thread.id,
                    error = %e,
                    orchestrator_version,
                    "orchestrator execution failed"
                );

                // Record failure for auto-rollback tracking
                if let Some(store) = self.store.as_ref() {
                    crate::executor::orchestrator::record_orchestrator_failure(
                        store,
                        self.thread.project_id,
                        orchestrator_version,
                    )
                    .await;

                    // Emit rollback event if this version will be skipped next time
                    // (failure count was just incremented, so check >= threshold - 1)
                    if orchestrator_version > 0 {
                        self.emit_event(EventKind::OrchestratorRollback {
                            from_version: orchestrator_version,
                            to_version: orchestrator_version.saturating_sub(1),
                            reason: format!("execution failed: {e}"),
                        });
                    }
                }

                // Transition to failed if not already in a terminal state
                if self.thread.state != ThreadState::Completed
                    && self.thread.state != ThreadState::Failed
                    && self.thread.state != ThreadState::Done
                {
                    let _ = self.thread.transition_to(
                        ThreadState::Failed,
                        Some(format!("orchestrator error: {e}")),
                    );
                }
                self.clear_runtime_checkpoint();
                self.persist_runtime_state(None, &mut persisted_event_count)
                    .await?;
                Ok(ThreadOutcome::Failed {
                    error: format!("Orchestrator error: {e}"),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::effect::ThreadExecutionContext;
    use crate::traits::llm::{LlmCallConfig, LlmOutput};
    use crate::types::capability::{ActionDef, CapabilityLease, EffectType, GrantedActions};
    use crate::types::project::ProjectId;
    use crate::types::step::LlmResponse;
    use crate::types::step::{ActionResult, TokenUsage};
    use crate::types::thread::{ThreadConfig, ThreadType};

    use crate::runtime::messaging::ThreadSignal;

    use std::sync::Mutex;
    use std::time::Duration;

    // ── Mock LLM ────────────────────────────────────────────

    struct MockLlm {
        responses: Mutex<Vec<LlmOutput>>,
    }

    impl MockLlm {
        fn new(responses: Vec<LlmOutput>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for MockLlm {
        async fn complete(
            &self,
            _messages: &[ThreadMessage],
            _actions: &[ActionDef],
            _config: &LlmCallConfig,
        ) -> Result<LlmOutput, EngineError> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(LlmOutput {
                    response: LlmResponse::Code {
                        code: "FINAL('(no more responses)')".into(),
                        content: Some("```repl\nFINAL('(no more responses)')\n```".into()),
                    },
                    usage: TokenUsage::default(),
                })
            } else {
                Ok(responses.remove(0))
            }
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    // ── Mock EffectExecutor ─────────────────────────────────

    struct MockEffects {
        results: Mutex<Vec<Result<ActionResult, EngineError>>>,
        actions: Vec<ActionDef>,
    }

    impl MockEffects {
        fn new(actions: Vec<ActionDef>, results: Vec<Result<ActionResult, EngineError>>) -> Self {
            Self {
                results: Mutex::new(results),
                actions,
            }
        }
    }

    #[async_trait::async_trait]
    impl EffectExecutor for MockEffects {
        async fn execute_action(
            &self,
            _action_name: &str,
            _parameters: serde_json::Value,
            _lease: &CapabilityLease,
            _context: &ThreadExecutionContext,
        ) -> Result<ActionResult, EngineError> {
            let mut results = self.results.lock().unwrap();
            if results.is_empty() {
                Ok(ActionResult {
                    call_id: String::new(),
                    action_name: String::new(),
                    output: serde_json::json!({"result": "ok"}),
                    is_error: false,
                    duration: Duration::from_millis(1),
                })
            } else {
                results.remove(0)
            }
        }

        async fn available_actions(
            &self,
            _leases: &[CapabilityLease],
        ) -> Result<Vec<ActionDef>, EngineError> {
            Ok(self.actions.clone())
        }
    }

    // ── Helpers ─────────────────────────────────────────────

    fn test_action() -> ActionDef {
        ActionDef {
            name: "test_tool".into(),
            description: "A test tool".into(),
            parameters_schema: serde_json::json!({"type": "object"}),
            effects: vec![EffectType::ReadLocal],
            requires_approval: false,
        }
    }

    async fn make_loop(
        llm_responses: Vec<LlmOutput>,
        effect_results: Vec<Result<ActionResult, EngineError>>,
        config: ThreadConfig,
    ) -> (ExecutionLoop, crate::runtime::messaging::SignalSender) {
        let project_id = ProjectId::new();
        let thread = Thread::new(
            "test goal",
            ThreadType::Foreground,
            project_id,
            "test-user",
            config,
        );
        let tid = thread.id;

        let llm = Arc::new(MockLlm::new(llm_responses));
        let effects = Arc::new(MockEffects::new(vec![test_action()], effect_results));
        let leases = Arc::new(LeaseManager::new());
        let policy = Arc::new(PolicyEngine::new());

        // Grant a default lease
        leases
            .grant(tid, "test_cap", GrantedActions::All, None, None)
            .await
            .unwrap();

        let (tx, rx) = crate::runtime::messaging::signal_channel(16);

        let exec = ExecutionLoop::new(thread, llm, effects, leases, policy, rx, "test-user".into());
        (exec, tx)
    }

    // ── Tests ───────────────────────────────────────────────

    fn code_response(code: &str) -> LlmOutput {
        LlmOutput {
            response: LlmResponse::Code {
                code: code.into(),
                content: Some(format!("```repl\n{code}\n```")),
            },
            usage: TokenUsage {
                input_tokens: 100,
                output_tokens: 80,
                ..Default::default()
            },
        }
    }

    // ── Loop lifecycle tests ────────────────────────────────

    /// Loop must exit with MaxIterations when the LLM never emits FINAL.
    /// Safety invariant — without this guard, a misbehaving model could
    /// burn the entire token budget in an endless loop.
    #[tokio::test]
    async fn max_iterations_reached() {
        let responses: Vec<LlmOutput> = (0..5)
            .map(|i| code_response(&format!("x = {i}")))
            .collect();

        let config = ThreadConfig {
            max_iterations: 3,
            ..ThreadConfig::default()
        };

        let (mut exec, _tx) = make_loop(responses, vec![], config).await;

        let outcome = exec.run().await.unwrap();
        assert!(matches!(
            outcome,
            ThreadOutcome::MaxIterations | ThreadOutcome::Completed { .. }
        ));
        assert!(exec.thread.step_count <= 3);
    }

    /// A stop signal mid-loop must interrupt execution and yield Stopped.
    #[tokio::test]
    async fn stop_signal_exits() {
        // 100 no-FINAL code responses — loop would run forever without stop.
        let responses: Vec<LlmOutput> = (0..100)
            .map(|i| code_response(&format!("x = {i}")))
            .collect();

        let (mut exec, tx) = make_loop(responses, vec![], ThreadConfig::default()).await;

        tx.send(ThreadSignal::Stop).await.unwrap();

        let outcome = exec.run().await.unwrap();
        assert!(matches!(outcome, ThreadOutcome::Stopped));
    }

    /// A message injected via ThreadSignal::InjectMessage must appear in
    /// the thread transcript so subsequent iterations see it.
    #[tokio::test]
    async fn inject_message_appears_in_context() {
        let (mut exec, tx) = make_loop(
            vec![code_response("FINAL('Got your message')")],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        tx.send(ThreadSignal::InjectMessage(ThreadMessage::user(
            "injected!",
        )))
        .await
        .unwrap();

        let outcome = exec.run().await.unwrap();
        assert!(matches!(outcome, ThreadOutcome::Completed { .. }));
        assert!(
            exec.thread
                .messages
                .iter()
                .chain(exec.thread.internal_messages.iter())
                .any(|m| m.content == "injected!"),
            "injected message should appear in thread transcript",
        );
    }

    /// Lifecycle events must be emitted: at minimum StateChanged transitions
    /// and StepStarted/StepCompleted pairs. Observability invariant.
    #[tokio::test]
    async fn events_are_recorded() {
        let (mut exec, _tx) = make_loop(
            vec![code_response("FINAL('Hello!')")],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        exec.run().await.unwrap();

        assert!(exec.thread.events.len() >= 4);

        // First event must be StateChanged(Created -> Running).
        assert!(matches!(
            &exec.thread.events[0].kind,
            EventKind::StateChanged {
                from: ThreadState::Created,
                to: ThreadState::Running,
                ..
            }
        ));
    }

    // ── call_id propagation through the code path ──────────
    //
    // Tool calls dispatched from inside Python (via Monty host functions)
    // still flow through the same EffectExecutor pipeline. Recorded
    // ActionResult messages and ActionExecuted events MUST carry non-empty
    // call_ids so the LLM API doesn't reject subsequent requests with
    // "No tool output found for function call <id>".


    /// Code-path tool calls must produce ActionExecuted events with
    /// non-empty call_ids. Tool calls are async — they must be awaited.
    #[tokio::test]
    async fn action_executed_events_carry_call_id() {
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "result = await test_tool()\nFINAL('ok')",
            )],
            vec![Ok(ActionResult {
                call_id: String::new(), // EffectExecutor returns empty — engine must fill
                action_name: "test_tool".into(),
                output: serde_json::json!({"data": "result"}),
                is_error: false,
                duration: Duration::from_millis(5),
            })],
            ThreadConfig::default(),
        )
        .await;

        exec.run().await.unwrap();

        let exec_events: Vec<_> = exec
            .thread
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::ActionExecuted { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();

        assert!(
            !exec_events.is_empty(),
            "should have at least one ActionExecuted event after awaiting test_tool()"
        );
        for call_id in &exec_events {
            assert!(
                !call_id.is_empty(),
                "ActionExecuted event must have non-empty call_id"
            );
        }
    }

    /// When a code-path tool call returns an error, the recorded
    /// ActionFailed event must carry a non-empty call_id. This keeps
    /// traces correlatable and prevents the downstream LLM API from
    /// rejecting the sequence with "No tool output found for function
    /// call <id>".
    #[tokio::test]
    async fn failed_action_preserves_call_id_in_message_and_event() {
        // MockEffects returns an is_error=true result; scripting.rs surfaces
        // this as ActionFailed via resolve_tool_future:1544. Lease denial
        // isn't a viable path here because `reconcile_dynamic_tool_lease`
        // auto-grants a "tools" lease covering every available action.
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "result = await test_tool()\nFINAL('done')",
            )],
            vec![Ok(ActionResult {
                call_id: String::new(),
                action_name: "test_tool".into(),
                output: serde_json::json!({"error": "simulated tool failure"}),
                is_error: true,
                duration: Duration::from_millis(1),
            })],
            ThreadConfig::default(),
        )
        .await;

        exec.run().await.unwrap();

        let fail_events: Vec<_> = exec
            .thread
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::ActionFailed { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect();

        assert!(
            !fail_events.is_empty(),
            "expected at least one ActionFailed event when the tool returned is_error=true"
        );
        for call_id in &fail_events {
            assert!(!call_id.is_empty(), "ActionFailed event must have call_id");
        }
    }

    // ── CodeAct / RLM tests ─────────────────────────────────

    #[tokio::test]
    async fn codeact_simple_final() {
        // LLM outputs Python code that calls FINAL()
        let (mut exec, _tx) = make_loop(
            vec![code_response("FINAL('The answer is 42')")],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "The answer is 42")
        );
        assert_eq!(exec.thread.step_count, 1);
    }

    #[tokio::test]
    async fn codeact_tool_call_then_final() {
        // LLM outputs code that calls a tool, then uses the result
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "result = test_tool()\nprint(result)\nFINAL('got result')",
            )],
            vec![Ok(ActionResult {
                call_id: "code_call_1".into(),
                action_name: "test_tool".into(),
                output: serde_json::json!({"data": "hello from tool"}),
                is_error: false,
                duration: Duration::from_millis(5),
            })],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "got result")
        );
        // Should have at least 1 action result recorded
        assert!(!exec.thread.internal_messages.is_empty());
    }

    #[tokio::test]
    async fn codeact_pure_python_computation() {
        // LLM outputs pure Python with no tool calls — just computation + FINAL
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "numbers = [1, 2, 3, 4, 5]\ntotal = sum(numbers)\nFINAL(f'Sum is {total}')",
            )],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "Sum is 15")
        );
    }

    #[tokio::test]
    async fn codeact_multi_step() {
        // First iteration: code runs but no FINAL — returns output
        // Second iteration: LLM sees output and calls FINAL
        let (mut exec, _tx) = make_loop(
            vec![
                code_response("x = 10 + 20\nprint(f'x = {x}')"),
                code_response("FINAL('done, x was 30')"),
            ],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "done, x was 30")
        );
        assert_eq!(exec.thread.step_count, 2);
        // The first step's stdout/output metadata is persisted as part of
        // the orchestrator's working messages → `internal_messages`.
        assert!(
            exec.thread
                .internal_messages
                .iter()
                .any(|m| m.content.contains("x = 30"))
        );
    }

    #[tokio::test]
    async fn codeact_error_recovery() {
        // First iteration: code has an error (NameError)
        // Second iteration: LLM sees the error and fixes it
        let (mut exec, _tx) = make_loop(
            vec![
                code_response("result = undefined_var + 1"),
                code_response("FINAL('recovered')"),
            ],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "recovered")
        );
        assert_eq!(exec.thread.step_count, 2);
        // First step should have error in output metadata
        assert!(
            exec.thread
                .internal_messages
                .iter()
                .any(|m| { m.content.contains("NameError") || m.content.contains("Error") })
        );
    }

    #[tokio::test]
    async fn codeact_context_variables_available() {
        // Code accesses the `goal` and `context` variables injected by the engine
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "FINAL(f'Goal: {goal}, Messages: {len(context)}')",
            )],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        // Should have access to goal="test goal" and context (list of messages)
        match outcome {
            ThreadOutcome::Completed { response: Some(r) } => {
                assert!(r.contains("Goal: test goal"), "got: {r}");
                assert!(r.contains("Messages:"), "got: {r}");
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn codeact_multiple_tool_calls_in_loop() {
        // Code calls a tool 3 times in a for loop
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "results = []\nfor i in range(3):\n    r = test_tool()\n    results.append(r)\nFINAL(f'Got {len(results)} results')",
            )],
            vec![
                Ok(ActionResult {
                    call_id: "code_call_1".into(),
                    action_name: "test_tool".into(),
                    output: serde_json::json!({"i": 0}),
                    is_error: false,
                    duration: Duration::from_millis(1),
                }),
                Ok(ActionResult {
                    call_id: "code_call_2".into(),
                    action_name: "test_tool".into(),
                    output: serde_json::json!({"i": 1}),
                    is_error: false,
                    duration: Duration::from_millis(1),
                }),
                Ok(ActionResult {
                    call_id: "code_call_3".into(),
                    action_name: "test_tool".into(),
                    output: serde_json::json!({"i": 2}),
                    is_error: false,
                    duration: Duration::from_millis(1),
                }),
            ],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        assert!(
            matches!(outcome, ThreadOutcome::Completed { response: Some(r) } if r == "Got 3 results")
        );
    }

    #[tokio::test]
    async fn codeact_llm_query_recursive() {
        // Code calls llm_query() — which calls the MockLlm recursively.
        // The MockLlm will return the next response in its queue for the sub-call.
        let (mut exec, _tx) = make_loop(
            vec![
                // First response: code that calls llm_query
                code_response(
                    "answer = llm_query('What is 2+2?')\nFINAL(f'Sub-agent said: {answer}')",
                ),
                // This text response will be consumed by the llm_query sub-call
                // (MockLlm pops from the same queue)
            ],
            vec![],
            ThreadConfig::default(),
        )
        .await;

        let outcome = exec.run().await.unwrap();
        // llm_query will get "(no more responses)" since the queue only had
        // the code response. That's fine — it tests the plumbing.
        match outcome {
            ThreadOutcome::Completed { response: Some(r) } => {
                assert!(r.contains("Sub-agent said:"), "got: {r}");
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    /// Verify the trace analyzer does NOT flag any issues on a clean
    /// code-path execution (no empty call_ids).
    #[tokio::test]
    async fn trace_analysis_clean_after_successful_tool_use() {
        let (mut exec, _tx) = make_loop(
            vec![code_response(
                "result = test_tool()\nFINAL('All done')",
            )],
            vec![Ok(ActionResult {
                call_id: String::new(),
                action_name: "test_tool".into(),
                output: serde_json::json!({"status": "ok"}),
                is_error: false,
                duration: Duration::from_millis(3),
            })],
            ThreadConfig::default(),
        )
        .await;

        exec.run().await.unwrap();

        let trace = crate::executor::trace::build_trace(&exec.thread);
        let empty_id_issues: Vec<_> = trace
            .issues
            .iter()
            .filter(|i| i.category == "empty_call_id")
            .collect();

        assert!(
            empty_id_issues.is_empty(),
            "clean execution should have no empty_call_id issues, got: {empty_id_issues:?}"
        );
    }
}

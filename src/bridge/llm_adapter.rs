//! LLM bridge adapter — wraps `LlmProvider` as `ironclaw_engine::LlmBackend`.

use std::sync::Arc;

use ironclaw_engine::{
    ActionDef, EngineError, LlmBackend, LlmCallConfig, LlmOutput, LlmResponse, ThreadMessage,
    TokenUsage,
};

use crate::llm::{ChatMessage, LlmProvider, Role, ToolCall, sanitize_tool_messages};

/// Wraps an existing `LlmProvider` to implement the engine's `LlmBackend` trait.
pub struct LlmBridgeAdapter {
    provider: Arc<dyn LlmProvider>,
    /// Optional cheaper provider for sub-calls (depth > 0).
    cheap_provider: Option<Arc<dyn LlmProvider>>,
}

impl LlmBridgeAdapter {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        cheap_provider: Option<Arc<dyn LlmProvider>>,
    ) -> Self {
        Self {
            provider,
            cheap_provider,
        }
    }

    fn provider_for_depth(&self, depth: u32) -> &Arc<dyn LlmProvider> {
        if depth > 0 {
            self.cheap_provider.as_ref().unwrap_or(&self.provider)
        } else {
            &self.provider
        }
    }
}

#[async_trait::async_trait]
impl LlmBackend for LlmBridgeAdapter {
    async fn complete(
        &self,
        messages: &[ThreadMessage],
        _actions: &[ActionDef],
        config: &LlmCallConfig,
    ) -> Result<LlmOutput, EngineError> {
        let provider = self.provider_for_depth(config.depth);

        // Convert messages
        let mut chat_messages: Vec<ChatMessage> = messages.iter().map(thread_msg_to_chat).collect();
        sanitize_tool_messages(&mut chat_messages);

        let max_tokens = config.max_tokens.unwrap_or(4096);
        let temperature = config.temperature.unwrap_or(0.7);

        // Plain completion, no API tools field. Actions are listed in the
        // system prompt; the model writes Python that calls them. The response
        // is always treated as Python code — Monty is the only validator.
        let mut request = crate::llm::CompletionRequest::new(chat_messages)
            .with_max_tokens(max_tokens)
            .with_temperature(temperature);
        request.metadata = config.metadata.clone();
        if let Some(ref model) = config.model {
            request.model = Some(model.clone());
        }

        let response = provider
            .complete(request)
            .await
            .map_err(|e| EngineError::Llm {
                reason: e.to_string(),
            })?;

        // Two shapes:
        // - Callers that explicitly want prose (compaction, llm_query, etc.)
        //   set force_text=true. Return the response verbatim as Text so an
        //   incidental ```repl fence in the prose isn't mistakenly extracted
        //   as "the answer".
        // - Everyone else is on the CodeAct path: extract a fenced block if
        //   present, else treat the whole response as code. Monty surfaces
        //   SyntaxError / NameError etc. naturally.
        let llm_response = if config.force_text {
            LlmResponse::Text(response.content)
        } else {
            let code = extract_code_block(&response.content)
                .unwrap_or_else(|| response.content.trim().to_string());
            LlmResponse::Code {
                code,
                content: Some(response.content),
            }
        };

        Ok(LlmOutput {
            response: llm_response,
            usage: TokenUsage {
                input_tokens: u64::from(response.input_tokens),
                output_tokens: u64::from(response.output_tokens),
                cache_read_tokens: u64::from(response.cache_read_input_tokens),
                cache_write_tokens: u64::from(response.cache_creation_input_tokens),
                cost_usd: 0.0,
            },
        })
    }

    fn model_name(&self) -> &str {
        self.provider.model_name()
    }
}

// ── Conversion helpers ──────────────────────────────────────

fn thread_msg_to_chat(msg: &ThreadMessage) -> ChatMessage {
    use ironclaw_engine::MessageRole;

    let role = match msg.role {
        MessageRole::System => Role::System,
        MessageRole::User => Role::User,
        MessageRole::Assistant => Role::Assistant,
        MessageRole::ActionResult => Role::Tool,
    };

    let mut chat = ChatMessage {
        role,
        content: msg.content.clone(),
        content_parts: Vec::new(),
        tool_call_id: msg.action_call_id.clone(),
        name: msg.action_name.clone(),
        tool_calls: None,
    };

    // Convert action calls if present (assistant message with tool calls)
    if let Some(ref calls) = msg.action_calls {
        chat.tool_calls = Some(
            calls
                .iter()
                .map(|c| ToolCall {
                    id: c.id.clone(),
                    name: c.action_name.clone(),
                    arguments: c.parameters.clone(),
                    reasoning: None,
                })
                .collect(),
        );
    }

    chat
}

/// Extract Python code from fenced code blocks in the LLM response.
///
/// Tries these markers in order: ```repl, ```python, ```py, then bare ```
/// (if the content looks like Python). Collects ALL code blocks in the
/// response and concatenates them (models sometimes split code across
/// multiple blocks with explanation text between them).
///
/// Return values:
/// - `Some(code)` with non-empty `code` — at least one non-empty fence found.
/// - `Some("")` — at least one matching fence found, but they were all empty.
///   The caller should treat this as a no-op (empty script), NOT fall back
///   to executing the raw response (which would include the fence markers
///   themselves and SyntaxError).
/// - `None` — no matching fence at all. Caller decides the fallback.
fn extract_code_block(text: &str) -> Option<String> {
    let mut all_code = Vec::new();
    let mut saw_any_fence = false;

    // Try specific markers first, then bare backticks
    for marker in ["```repl", "```python", "```py", "```"] {
        let mut search_from = 0;
        // Only tracked for explicit-language markers (```repl / ```python /
        // ```py). Bare ``` blocks that fail `looks_like_python` below are
        // meant to be ignored entirely, not counted as "empty fence" — those
        // are usually markdown lists or tables in the model's narrative.
        let mut saw_typed_fence = false;
        while let Some(start) = text[search_from..].find(marker) {
            let abs_start = search_from + start;
            let after_marker = abs_start + marker.len();

            // For bare ```, skip if it's actually ```someotherlang
            if marker == "```" && text[after_marker..].starts_with(|c: char| c.is_alphabetic()) {
                let lang: String = text[after_marker..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect();
                if !["repl", "python", "py"].contains(&lang.as_str()) {
                    search_from = after_marker;
                    continue;
                }
            }

            // Skip to next line after the marker
            let code_start = text[after_marker..]
                .find('\n')
                .map(|i| after_marker + i + 1)
                .unwrap_or(after_marker);

            // Find closing ```
            if let Some(end) = text[code_start..].find("```") {
                let code = text[code_start..code_start + end].trim();
                if !code.is_empty() {
                    // For bare ``` blocks (no explicit language tag) only
                    // accept content that actually looks like Python. Without
                    // this guard, the agent's example markdown blocks
                    // (lists, tables, plain prose) get misclassified as code
                    // and explode in the Monty parser with SyntaxError —
                    // which the LLM then has to recover from.
                    if marker == "```" && !looks_like_python(code) {
                        search_from = code_start + end + 3;
                        continue;
                    }
                    all_code.push(code.to_string());
                } else if marker != "```" {
                    // Empty ```repl / ```python / ```py block — intentional
                    // no-op from the model, not bare markdown.
                    saw_typed_fence = true;
                }
                search_from = code_start + end + 3;
            } else {
                break;
            }
        }

        if saw_typed_fence {
            saw_any_fence = true;
        }
        // If we found code with a specific marker, use it (don't fall through to bare)
        if !all_code.is_empty() {
            break;
        }
    }

    if !all_code.is_empty() {
        Some(all_code.join("\n\n"))
    } else if saw_any_fence {
        // Fences exist but all empty — return empty string so the caller
        // doesn't fall back to the raw response (which would hand the fence
        // markers themselves to Monty as "code").
        Some(String::new())
    } else {
        None
    }
}

/// Heuristic check that a bare ``` block contains Python rather than
/// markdown / prose / a different language.
///
/// Accepts: assignments (`x =`), function calls (`name(`), Python keywords
/// (`import`, `from`, `def`, `class`, `if`, `for`, `while`, `return`,
/// `print`, `FINAL`, `try`, `with`, `pass`, `raise`, `yield`, `lambda`),
/// or comments (`#`).
///
/// Rejects: lines starting with `-`, `*`, `|`, `>`, `:`, digits followed by
/// `.` (markdown lists, tables, blockquotes, headings, numbered lists),
/// bare prose, etc.
/// Returns true when `line` contains an identifier-style function call
/// (an identifier or attribute path immediately followed by `(`).
///
/// Avoids the false positives `trimmed.contains('(')` produced for markdown
/// links like `[text](url)` and prose like "See (docs)" — neither has an
/// alphanumeric/underscore character directly before the `(`.
fn has_identifier_call(line: &str) -> bool {
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'(' && i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                return true;
            }
        }
    }
    false
}

fn looks_like_python(code: &str) -> bool {
    const PY_KEYWORDS: &[&str] = &[
        "import", "from", "def", "class", "if", "for", "while", "return", "print", "FINAL", "try",
        "with", "pass", "raise", "yield", "lambda", "elif", "else", "async", "await", "global",
        "nonlocal", "assert", "break", "continue", "del", "not", "and", "or", "is", "in",
    ];

    // Check the first few non-empty lines — at least one must look Python-y.
    for line in code.lines().take(5) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Comments are valid Python.
        if trimmed.starts_with('#') {
            return true;
        }
        // Markdown markers are NOT Python.
        if trimmed.starts_with('-')
            || trimmed.starts_with('*')
            || trimmed.starts_with('|')
            || trimmed.starts_with('>')
        {
            return false;
        }
        // Markdown numbered list "1. foo" is NOT Python (a Python statement
        // starting with a literal int is `123` followed by an operator, not
        // `123. text`).
        if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) && trimmed.contains(". ") {
            return false;
        }
        // Function call: an identifier (or attribute path) followed by `(`,
        // e.g. `foo(...)`, `obj.method(...)`. We require the `(` to be
        // preceded by an identifier char so markdown links like `[text](url)`
        // and prose like "See (docs)" don't get classified as Python.
        if has_identifier_call(trimmed) {
            return true;
        }
        // Assignment: `name = ...` (but not `==` comparisons in prose).
        if trimmed.contains('=') {
            return true;
        }
        // First word matches a Python keyword.
        let first_word: String = trimmed
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if PY_KEYWORDS.contains(&first_word.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use rust_decimal::Decimal;

    use ironclaw_engine::{ActionCall, ActionDef, EffectType, LlmResponse, ThreadMessage};

    use crate::error::LlmError;
    use crate::llm::{ToolCompletionRequest, ToolCompletionResponse};

    #[derive(Default)]
    struct CapturingProviderState {
        completion_requests: tokio::sync::Mutex<Vec<Vec<ChatMessage>>>,
        models: tokio::sync::Mutex<Vec<Option<String>>>,
    }

    struct CapturingProvider {
        state: Arc<CapturingProviderState>,
    }

    #[async_trait]
    impl LlmProvider for CapturingProvider {
        fn model_name(&self) -> &str {
            "capturing-provider"
        }

        fn cost_per_token(&self) -> (Decimal, Decimal) {
            (Decimal::ZERO, Decimal::ZERO)
        }

        async fn complete(
            &self,
            req: crate::llm::CompletionRequest,
        ) -> Result<crate::llm::CompletionResponse, LlmError> {
            self.state.models.lock().await.push(req.model.clone());
            self.state
                .completion_requests
                .lock()
                .await
                .push(req.messages);

            Ok(crate::llm::CompletionResponse {
                content: "ok".to_string(),
                input_tokens: 1,
                output_tokens: 1,
                finish_reason: crate::llm::FinishReason::Stop,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        }

        // The adapter never calls this under the code-only contract, but the
        // LlmProvider trait still requires it.
        async fn complete_with_tools(
            &self,
            _req: ToolCompletionRequest,
        ) -> Result<ToolCompletionResponse, LlmError> {
            unreachable!("adapter should never call complete_with_tools under code-only")
        }
    }

    #[tokio::test]
    async fn complete_rewrites_orphaned_action_results_before_provider_call() {
        let state = Arc::new(CapturingProviderState::default());
        let provider: Arc<dyn LlmProvider> = Arc::new(CapturingProvider {
            state: state.clone(),
        });
        let adapter = LlmBridgeAdapter::new(provider, None);
        let messages = vec![
            ThreadMessage::user("Find the docs"),
            ThreadMessage::assistant("I checked a tool earlier."),
            ThreadMessage::action_result("call_missing", "search", "result payload"),
        ];

        let output = adapter
            .complete(&messages, &[], &LlmCallConfig::default())
            .await
            .unwrap();

        match output.response {
            LlmResponse::Code { ref code, .. } => assert_eq!(code, "ok"),
            other => panic!("expected code response, got {other:?}"),
        }

        let completion_requests = state.completion_requests.lock().await;
        let sent = completion_requests.last().unwrap();

        assert_eq!(sent.len(), 3);
        assert_eq!(sent[2].role, Role::User);
        assert_eq!(sent[2].content, "[Tool `search` returned: result payload]");
        assert!(sent[2].tool_call_id.is_none());
        assert!(sent[2].name.is_none());
    }

    #[tokio::test]
    async fn config_model_forwards_to_completion_request() {
        let state = Arc::new(CapturingProviderState::default());
        let provider: Arc<dyn LlmProvider> = Arc::new(CapturingProvider {
            state: state.clone(),
        });
        let adapter = LlmBridgeAdapter::new(provider, None);

        let config = ironclaw_engine::LlmCallConfig {
            model: Some("gpt-4o".into()),
            ..Default::default()
        };

        // Actions are now ignored at the API boundary — the adapter always
        // uses plain completion and the model writes Python that references
        // actions by name.
        adapter
            .complete(
                &[ThreadMessage::user("hi")],
                &[ActionDef {
                    name: "echo".into(),
                    description: "test".into(),
                    parameters_schema: serde_json::json!({"type": "object"}),
                    effects: vec![EffectType::ReadLocal],
                    requires_approval: false,
                }],
                &config,
            )
            .await
            .unwrap();

        let models = state.models.lock().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].as_deref(), Some("gpt-4o"));
    }

    /// Regression for compaction data loss (reviewer finding #1 + #3).
    ///
    /// Callers that want prose (compaction, `llm_query`, `llm_query_batched`)
    /// set `force_text=true`. The adapter MUST return `LlmResponse::Text`
    /// with the full content verbatim — not a `Code` variant where the
    /// model's prose gets lost and a `code` field extracted from an
    /// incidental ```repl fence becomes "the answer".
    #[tokio::test]
    async fn force_text_returns_text_verbatim_including_incidental_fence() {
        struct ProseWithFenceProvider;

        #[async_trait]
        impl LlmProvider for ProseWithFenceProvider {
            fn model_name(&self) -> &str {
                "prose-fence-mock"
            }
            fn cost_per_token(&self) -> (Decimal, Decimal) {
                (Decimal::ZERO, Decimal::ZERO)
            }
            async fn complete(
                &self,
                _req: crate::llm::CompletionRequest,
            ) -> Result<crate::llm::CompletionResponse, LlmError> {
                // A realistic compaction summary that quotes the agent's
                // prior code for context. The fence is incidental content,
                // not the answer.
                Ok(crate::llm::CompletionResponse {
                    content: "Summary: the agent ran `x = 1` as\n\
                              ```repl\n\
                              x = 1\n\
                              print(x)\n\
                              ```\n\
                              Progress: 1 of 3 steps done."
                        .to_string(),
                    input_tokens: 5,
                    output_tokens: 5,
                    finish_reason: crate::llm::FinishReason::Stop,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                })
            }
            async fn complete_with_tools(
                &self,
                _req: ToolCompletionRequest,
            ) -> Result<ToolCompletionResponse, LlmError> {
                unreachable!("force_text path should not call complete_with_tools")
            }
        }

        let provider: Arc<dyn LlmProvider> = Arc::new(ProseWithFenceProvider);
        let adapter = LlmBridgeAdapter::new(provider, None);

        let config = ironclaw_engine::LlmCallConfig {
            force_text: true,
            ..Default::default()
        };

        let output = adapter
            .complete(&[ThreadMessage::user("summarize")], &[], &config)
            .await
            .unwrap();

        match output.response {
            LlmResponse::Text(text) => {
                assert!(text.contains("Summary:"), "prose preamble lost: {text}");
                assert!(text.contains("Progress:"), "prose suffix lost: {text}");
                assert!(
                    text.contains("```repl"),
                    "incidental fence stripped from prose: {text}"
                );
            }
            other => panic!("expected LlmResponse::Text under force_text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn config_without_model_leaves_request_model_none() {
        let state = Arc::new(CapturingProviderState::default());
        let provider: Arc<dyn LlmProvider> = Arc::new(CapturingProvider {
            state: state.clone(),
        });
        let adapter = LlmBridgeAdapter::new(provider, None);

        adapter
            .complete(
                &[ThreadMessage::user("hi")],
                &[],
                &ironclaw_engine::LlmCallConfig::default(),
            )
            .await
            .unwrap();

        let models = state.models.lock().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0], None);
    }

    // ── extract_code_block tests ────────────────────────────

    #[test]
    fn extract_repl_block() {
        let text = "Some explanation\n```repl\nx = 1 + 2\nprint(x)\n```\nMore text";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code, "x = 1 + 2\nprint(x)");
    }

    #[test]
    fn extract_python_block() {
        let text = "Let me compute:\n```python\nresult = sum([1,2,3])\n```";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code, "result = sum([1,2,3])");
    }

    #[test]
    fn extract_py_block() {
        let text = "```py\nprint('hello')\n```";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code, "print('hello')");
    }

    #[test]
    fn extract_bare_backtick_block() {
        // Bare ``` blocks are accepted ONLY when the content looks like
        // Python (assignment, function call, keyword, or comment). The
        // `looks_like_python` heuristic prevents the LLM's example markdown
        // from being misclassified as code (which used to crash Monty
        // with a SyntaxError on `- TICKER: SIZE, ...` style content).
        let text = "Here's the code:\n```\nx = 42\nFINAL(x)\n```";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code, "x = 42\nFINAL(x)");
    }

    #[test]
    fn bare_backtick_markdown_list_is_rejected() {
        let text = "Example positions file:\n```\n- AAPL: 500 shares, entry $175\n- TSLA: 200 shares, entry $260\n```";
        assert!(
            extract_code_block(text).is_none(),
            "markdown list inside bare ``` should NOT be treated as Python"
        );
    }

    #[test]
    fn bare_backtick_markdown_table_is_rejected() {
        let text = "Schema:\n```\n| col | type |\n| --- | --- |\n| id  | int  |\n```";
        assert!(
            extract_code_block(text).is_none(),
            "markdown table inside bare ``` should NOT be treated as Python"
        );
    }

    #[test]
    fn bare_backtick_prose_is_rejected() {
        let text = "Here's a quote:\n```\nThe quick brown fox jumps over the lazy dog.\n```";
        assert!(
            extract_code_block(text).is_none(),
            "prose inside bare ``` should NOT be treated as Python"
        );
    }

    #[test]
    fn bare_backtick_markdown_link_is_rejected() {
        // Regression test for PR #1736 review (Copilot, 3057247912):
        // `looks_like_python` previously matched any line containing `(`,
        // which classified markdown links like `[text](url)` and prose
        // like "See (docs)" as Python and forwarded them to Monty as code.
        let link_text = "Read more:\n```\n[the docs](https://example.com)\n```";
        assert!(
            extract_code_block(link_text).is_none(),
            "markdown link inside bare ``` should NOT be treated as Python"
        );

        let parens_prose = "Note:\n```\nSee (docs) for details on the API.\n```";
        assert!(
            extract_code_block(parens_prose).is_none(),
            "prose with parenthetical inside bare ``` should NOT be treated as Python"
        );
    }

    #[test]
    fn bare_backtick_python_with_comment() {
        let text = "```\n# fetch the data\nresult = fetch()\nFINAL(result)\n```";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("fetch()"));
    }

    #[test]
    fn skip_non_python_language() {
        let text = "```json\n{\"key\": \"value\"}\n```\nThat's the config.";
        assert!(extract_code_block(text).is_none());
    }

    #[test]
    fn no_code_blocks_returns_none() {
        let text = "Just a plain text response with no code.";
        assert!(extract_code_block(text).is_none());
    }

    #[test]
    fn multiple_code_blocks_concatenated() {
        let text = "\
Let me search first:\n\
```repl\nresult = web_search(query=\"test\")\nprint(result)\n```\n\
Now let's process:\n\
```repl\nFINAL(result['title'])\n```";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("web_search"));
        assert!(code.contains("FINAL"));
        // Two blocks joined by double newline
        assert!(code.contains("\n\n"));
    }

    #[test]
    fn mixed_thinking_and_code() {
        // Simulates a model that outputs explanation + code (the Hyperliquid case)
        let text = "\
Let me help you explore the relationship between Hyperliquid's price and revenue.\n\
\n\
First, let's gather some data:\n\
\n\
```python\nsearch_results = web_search(\n    query=\"Hyperliquid revenue\",\n    count=5\n)\nprint(search_results)\n```\n\
\n\
And also check the token price:\n\
\n\
```python\ntoken_data = web_search(\n    query=\"Hyperliquid token price\",\n    count=3\n)\nprint(token_data)\n```";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("web_search"));
        assert!(code.contains("Hyperliquid revenue"));
        assert!(code.contains("Hyperliquid token price"));
    }

    #[test]
    fn repl_preferred_over_bare() {
        // If both ```repl and bare ``` exist, prefer ```repl
        let text = "```\nignored\n```\n```repl\nused = True\n```";
        let code = extract_code_block(text).unwrap();
        assert_eq!(code, "used = True");
    }

    #[test]
    fn empty_code_block_returns_empty_not_none() {
        // Empty fence → Some(""), so the adapter does NOT fall back to
        // handing the raw response (including fence markers) to Monty.
        // Regression for reviewer finding #5.
        let text = "```python\n\n```\nThat was empty.";
        assert_eq!(extract_code_block(text).as_deref(), Some(""));
    }

    #[test]
    fn unclosed_block_returns_none() {
        let text = "```python\nprint('no closing fence')";
        assert!(extract_code_block(text).is_none());
    }

    /// Regression test: the full ThreadMessage -> ChatMessage -> sanitize
    /// pipeline must preserve 1:1 correspondence between assistant
    /// tool_calls and Tool messages. A gap causes the LLM API to reject
    /// with "No tool output found for function call <id>".
    #[test]
    fn tool_call_result_correspondence_after_sanitize() {
        // Simulate messages that include a "[no output]" placeholder
        // (the fix for null tool output).
        let messages: Vec<ThreadMessage> = vec![
            ThreadMessage::system("system prompt"),
            ThreadMessage::user("update all tools"),
            ThreadMessage::assistant_with_actions(
                Some(String::new()),
                vec![
                    ActionCall {
                        id: "call_AAA".into(),
                        action_name: "tool_a".into(),
                        parameters: serde_json::json!({}),
                    },
                    ActionCall {
                        id: "call_BBB".into(),
                        action_name: "tool_b".into(),
                        parameters: serde_json::json!({}),
                    },
                    ActionCall {
                        id: "call_CCC".into(),
                        action_name: "tool_c".into(),
                        parameters: serde_json::json!({}),
                    },
                ],
            ),
            ThreadMessage::action_result("call_AAA", "tool_a", "{\"ok\": true}"),
            // call_BBB had null output; Python now sends "[no output]"
            ThreadMessage::action_result("call_BBB", "tool_b", "[no output]"),
            ThreadMessage::action_result("call_CCC", "tool_c", "{\"done\": true}"),
        ];

        let mut chat_messages: Vec<ChatMessage> = messages.iter().map(thread_msg_to_chat).collect();
        sanitize_tool_messages(&mut chat_messages);

        // Collect tool_call IDs from assistant messages
        let mut expected_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in &chat_messages {
            if msg.role == Role::Assistant
                && let Some(ref calls) = msg.tool_calls
            {
                for tc in calls {
                    expected_ids.insert(tc.id.clone());
                }
            }
        }

        // Collect tool_call_ids from Tool messages
        let mut result_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for msg in &chat_messages {
            if msg.role == Role::Tool
                && let Some(ref id) = msg.tool_call_id
            {
                result_ids.insert(id.clone());
            }
        }

        assert_eq!(expected_ids.len(), 3, "assistant should have 3 tool calls");
        for id in &expected_ids {
            assert!(
                result_ids.contains(id),
                "tool_call {id} has no matching Tool message after sanitize — \
                 LLM API would reject with 'No tool output found'"
            );
        }
    }

}

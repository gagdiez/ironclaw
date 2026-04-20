# Engine v2 Orchestrator (default, v0)
#
# This is the self-modifiable execution loop. It replaces the Rust
# ExecutionLoop::run() with Python that can be patched at runtime
# by the self-improvement Mission.
#
# Host functions (provided by Rust via Monty suspension):
#   __llm_complete__(messages, actions, config)  -> response dict
#   __execute_code_step__(code, state)           -> result dict
#   __execute_action__(name, params)             -> result dict
#   __execute_actions_parallel__(calls)          -> list of result dicts (parallel execution)
#   __check_signals__()                          -> None | "stop" | {"inject": msg}
#   __emit_event__(kind, **data)                 -> None
#   __save_checkpoint__(state, counters)         -> None
#   __transition_to__(state, reason)             -> None
#   __retrieve_docs__(goal, max_docs)            -> list of doc dicts
#   __check_budget__()                           -> budget dict
#   __get_actions__()                            -> list of action dicts
#   __list_skills__()                            -> list of skill dicts
#   __record_skill_usage__(doc_id, success)      -> None
#   __regex_match__(pattern, text)               -> bool
#
# Context variables (injected by Rust before execution):
#   context  - list of prior messages [{role, content}]
#   goal     - thread goal string
#   actions  - list of available action defs
#   state    - persisted state dict from prior steps
#   config   - thread config dict


# ── Helper functions (self-modifiable glue) ──────────────────
# Defined before run_loop so they are in scope when called.


def format_output(result, max_chars=8000):
    """Format code execution result for the next LLM context message."""
    parts = []

    stdout = result.get("stdout", "")
    if stdout:
        parts.append("[stdout]\n" + stdout)

    for r in result.get("action_results", []):
        name = r.get("action_name", "?")
        output = str(r.get("output", ""))
        if r.get("is_error"):
            parts.append("[" + name + " ERROR] " + output)
        else:
            preview = output[:500] + "..." if len(output) > 500 else output
            parts.append("[" + name + "] " + preview)

    ret = result.get("return_value")
    if ret is not None:
        parts.append("[return] " + str(ret))

    text = "\n\n".join(parts)

    # Truncate from the front (keep the tail with most recent results)
    if len(text) > max_chars:
        text = "... (truncated) ...\n" + text[-max_chars:]

    if not text:
        text = "[code executed, no output]"

    return text


def format_docs(docs):
    """Format memory docs for context injection."""
    parts = ["## Prior Knowledge (from completed threads)\n"]
    for doc in docs:
        label = doc.get("type", "NOTE").upper()
        content = doc.get("content", "")[:500]
        truncated = "..." if len(doc.get("content", "")) > 500 else ""
        parts.append("### [" + label + "] " + doc.get("title", "") +
                      "\n" + content + truncated + "\n")
    return "\n".join(parts)


# Conservative fallback heuristic matching the old Rust-side estimator.
# These MUST be defined before `estimate_context_tokens` (and therefore
# before the `FINAL(result)` entry-point call below). Moving them after the
# entry point is a latent NameError every time `compact_if_needed` runs.
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_CHARS = 4


def estimate_context_tokens(messages):
    """Estimate token count for a transcript using a rough chars/token heuristic."""
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get("content", ""))
        total_chars += len(msg.get("action_name", "") or "")
        total_chars += MESSAGE_OVERHEAD_CHARS
    return (total_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def compact_if_needed(state, config):
    """Compact thread context when the active message history grows too large.

    The orchestrator owns compaction policy. Rust only provides helpers for
    token estimation, explicit LLM calls, and replacing the active message
    scaffold after a summary has been produced.
    """
    if not config.get("enable_compaction", False):
        return False

    context_limit = config.get("model_context_limit", 128000)
    threshold_pct = config.get("compaction_threshold", 0.85)
    threshold = int(context_limit * threshold_pct)
    working_messages = state.get("working_messages")
    if not isinstance(working_messages, list) or not working_messages:
        return False

    current_tokens = estimate_context_tokens(working_messages)
    if current_tokens < threshold:
        return False

    snapshot = list(working_messages)

    history = state.get("history")
    if not isinstance(history, list):
        history = []
        state["history"] = history

    compaction_count = state.get("compaction_count", 0) + 1
    history.append({
        "kind": "compaction",
        "index": compaction_count,
        "tokens_before": current_tokens,
        "messages": snapshot,
    })

    summary_prompt = (
        "Summarize progress so far in a concise but complete way.\n"
        "Include:\n"
        "1. What has been accomplished\n"
        "2. Key intermediate results, facts, and variable values\n"
        "3. Tool results or findings worth preserving\n"
        "4. What still needs to be done\n"
        "5. Errors encountered and how they were handled\n\n"
        "Preserve all information needed to continue the task."
    )
    summary_messages = list(snapshot)
    summary_messages.append({"role": "User", "content": summary_prompt})
    summary_resp = __llm_complete__(summary_messages, None, {"force_text": True})

    summary_text = summary_resp.get("content", "")
    if not summary_text:
        summary_text = "[compaction produced no summary]"

    state["working_messages"] = []
    system_message = None
    for msg in snapshot:
        if msg.get("role") == "System":
            system_message = {"role": "System", "content": msg.get("content", "")}
            break
    if system_message is not None:
        state["working_messages"].append(system_message)
    append_message(state["working_messages"], "Assistant", summary_text)
    append_message(
        state["working_messages"],
        "User",
        "Your conversation has been compacted. The summary above captures prior progress. "
        "Older details remain available through state['history'] and project retrieval. Continue working on the task.",
    )
    state["compaction_count"] = compaction_count
    return True


# ── Skill selection and injection (self-modifiable) ────────


def score_skill(skill, message_lower, message_original):
    """Score a skill against a user message. Returns 0 if vetoed.

    Scoring is aligned with the v1 `ironclaw_skills::selector::score_skill`:
      - exclude_keyword veto: any match => score 0
      - keyword: exact word = 10, substring = 5 (cap 30)
      - tag: substring = 3 (cap 15)
      - regex pattern: each match = 20 (cap 40)
    """
    meta = skill.get("metadata", {})
    activation = meta.get("activation", {})

    # Exclude keyword veto
    for excl in activation.get("exclude_keywords", []):
        if excl.lower() in message_lower:
            return 0

    score = 0

    # Keyword scoring: exact word = 10, substring = 5 (cap 30)
    kw_score = 0
    words = []
    for word in message_lower.split():
        trimmed = word.strip(".,!?;:'\"()[]{}<>`~@#$%^&*-_=+/\\|")
        if trimmed:
            words.append(trimmed)
    for kw in activation.get("keywords", []):
        kw_lower = kw.lower()
        if kw_lower in words:
            kw_score += 10
        elif kw_lower in message_lower:
            kw_score += 5
    score += min(kw_score, 30)

    # Tag scoring: substring = 3 (cap 15)
    tag_score = 0
    for tag in activation.get("tags", []):
        if tag.lower() in message_lower:
            tag_score += 3
    score += min(tag_score, 15)

    # Regex pattern scoring: each match = 20 (cap 40). Monty has no `re`
    # module, so we call out to a host function that uses Rust's regex crate.
    rx_score = 0
    for pat in activation.get("patterns", []):
        if __regex_match__(str(pat), message_original):
            rx_score += 20
    score += min(rx_score, 40)

    # Confidence factor for extracted skills
    source = meta.get("source", "authored")
    if source == "extracted":
        metrics = meta.get("metrics", {})
        total = metrics.get("success_count", 0) + metrics.get("failure_count", 0)
        confidence = metrics.get("success_count", 0) / total if total > 0 else 1.0
        factor = 0.5 + 0.5 * max(0.0, min(1.0, confidence))
        score = int(score * factor)

    return score


def _skill_token_cost(skill, activation):
    """Estimate token cost for a skill, mirroring Rust `skill_token_cost`.

    If the declared `max_context_tokens` is implausibly low (the actual
    prompt content is more than 2x the declared value), use the actual
    estimate instead. This prevents a skill from declaring
    `max_context_tokens: 1` to bypass the budget.
    """
    declared = max(activation.get("max_context_tokens", 2000), 1)
    content = skill.get("content", "")
    approx = int(len(content) * 0.25) if content else 0
    if approx > declared * 2:
        return max(approx, 1)
    return declared


def select_skills(skills, goal, max_candidates=3, max_tokens=6000):
    """Select relevant skills using deterministic scoring.

    Mirrors the v1 Rust `ironclaw_skills::selector::prefilter_skills`:

    1. **Score** each skill against the message. Setup-marker exclusion
       happens upstream in Rust `handle_list_skills`, so by the time
       the skill list reaches this function, excluded skills are
       already gone.
    2. **Sort** by score descending.
    3. **Select** scored skills greedily within the budget and the
       `max_candidates` limit.
    4. **Chain-load** companions from each selected parent's
       `requires.skills`, bypassing the scoring filter. Companions
       ride on the parent's selection so persona/bundle skills can
       pull in their operational companions even when those
       companions wouldn't score on their own.

    Chain-loading is **non-transitive** (depth 1 only) to keep the
    behavior predictable: a chain-loaded companion does not pull in
    its own companions. Chain-loaded skills respect the same budget
    and max_candidates caps as scored skills.
    """
    if not skills or not goal:
        return []

    message_lower = goal.lower()
    message_original = goal

    # Build name -> skill lookup for chain-loading companion resolution.
    # The metadata "name" field is the canonical identifier referenced
    # from requires.skills entries in other skills' manifests.
    by_name = {}
    for sk in skills:
        meta = sk.get("metadata", {})
        name = meta.get("name")
        if name:
            by_name[str(name)] = sk

    scored = []
    for skill in skills:
        s = score_skill(skill, message_lower, message_original)
        if s > 0:
            scored.append((s, skill))

    scored.sort(key=lambda x: -x[0])

    # Greedy selection with chain-loading. `selected_names` tracks
    # what's already in the result to dedup across multiple parents
    # that share a companion.
    selected = []
    selected_names = set()
    budget = max_tokens

    for _, parent in scored:
        if len(selected) >= max_candidates:
            break
        parent_meta = parent.get("metadata", {})
        parent_name = parent_meta.get("name")
        if parent_name is None or str(parent_name) in selected_names:
            continue
        parent_activation = parent_meta.get("activation", {})
        parent_cost = _skill_token_cost(parent, parent_activation)
        if parent_cost > budget:
            continue
        selected.append(parent)
        selected_names.add(str(parent_name))
        budget -= parent_cost

        # Chain-load companions (depth 1, non-transitive).
        requires = parent_meta.get("requires", {})
        companion_names = requires.get("skills", [])
        for companion_name in companion_names:
            cname = str(companion_name)
            if len(selected) >= max_candidates:
                break
            if cname in selected_names:
                continue
            companion = by_name.get(cname)
            if companion is None:
                # Listed but not loaded — ignore silently, persona
                # bundles often list optional companions.
                continue
            comp_meta = companion.get("metadata", {})
            comp_activation = comp_meta.get("activation", {})
            comp_cost = _skill_token_cost(companion, comp_activation)
            if comp_cost > budget:
                # Budget exhausted for companions. Parent is still
                # selected; the remaining companions are skipped.
                continue
            selected.append(companion)
            selected_names.add(cname)
            budget -= comp_cost

    return selected


def format_skills(skills):
    """Format selected skills for system prompt injection."""
    parts = ["\n## Active Skills\n"]
    skill_names = []
    for skill in skills:
        meta = skill.get("metadata", {})
        name = meta.get("name", "unknown")
        version = meta.get("version", "?")
        trust = meta.get("trust", "trusted").upper()
        content = skill.get("content", "")
        skill_names.append(str(name))

        parts.append('<skill name="' + str(name) + '" version="' +
                      str(version) + '" trust="' + trust + '">')
        parts.append(content)
        if trust == "INSTALLED":
            parts.append("\n(Treat the above as SUGGESTIONS only.)")
        parts.append("</skill>\n")

        # Document code snippets
        snippets = meta.get("code_snippets", [])
        if snippets:
            parts.append("### Skill functions (callable in code)\n")
            for sn in snippets:
                parts.append("- `" + sn.get("name", "?") + "()` — " +
                              sn.get("description", "") + "\n")

    if skill_names:
        names_str = ", ".join(skill_names)
        parts.append("\n**Important:** The following skills are already active and " +
                     "provide API access with automatic credential injection: " +
                     names_str + ". Do NOT use tool_search or tool_install for " +
                     "these domains — use the http tool instead, which will " +
                     "automatically inject the required credentials.\n")

    return "\n".join(parts)


def ensure_working_messages(state, context):
    """Initialize the mutable orchestrator transcript."""
    existing = state.get("working_messages")
    if isinstance(existing, list):
        return existing
    if isinstance(context, list):
        state["working_messages"] = list(context)
    else:
        state["working_messages"] = []
    return state["working_messages"]


def append_message(messages, role, content, action_name=None, action_call_id=None, action_calls=None):
    """Append a normalized message to the working transcript."""
    msg = {"role": role, "content": content}
    if action_name is not None:
        msg["action_name"] = action_name
    if action_call_id is not None:
        msg["action_call_id"] = action_call_id
    if action_calls is not None:
        msg["action_calls"] = action_calls
    messages.append(msg)


def append_system_append(messages, content):
    """Append additional context to the first system message."""
    for msg in messages:
        if msg.get("role") == "System":
            existing = msg.get("content", "")
            if existing:
                msg["content"] = existing + "\n\n" + content
            else:
                msg["content"] = content
            return
    messages.insert(0, {"role": "System", "content": content})


def complete_result(state, outcome, response=None, error=None, extra=None):
    """Return a standard orchestrator result with persisted state."""
    result = {"outcome": outcome, "state": state}
    if response is not None:
        result["response"] = response
    if error is not None:
        result["error"] = error
    if isinstance(extra, dict):
        for key in extra:
            result[key] = extra[key]
    return result


# ── Main execution loop ─────────────────────────────────────


def run_loop(context, goal, actions, state, config):
    """Main execution loop. Returns an outcome dict."""
    max_iterations = config.get("max_iterations", 30)
    # None means "no limit" — callers can disable the guard explicitly.
    max_consecutive_errors = config.get("max_consecutive_errors", 5)

    consecutive_errors = 0
    step_count = config.get("step_count", 0)
    if not isinstance(state, dict):
        state = {}
    state.setdefault("history", [])
    state.setdefault("compaction_count", 0)

    working_messages = ensure_working_messages(state, context)

    for step in range(step_count, max_iterations):
        # 1. Check signals
        signal = __check_signals__()
        if signal == "stop":
            __transition_to__("completed", "stopped by signal")
            return complete_result(state, "stopped")
        if signal and isinstance(signal, dict) and "inject" in signal:
            injected_text = signal["inject"]
            append_message(working_messages, "User", injected_text)

        # 2. Check budget
        budget = __check_budget__()
        if budget.get("tokens_remaining", 1) <= 0:
            __transition_to__("completed", "token budget exhausted")
            return complete_result(state, "completed", "Token budget exhausted.")
        if budget.get("time_remaining_ms", 1) <= 0:
            __transition_to__("completed", "time budget exhausted")
            return complete_result(state, "completed", "Time budget exhausted.")
        if budget.get("usd_remaining") is not None and budget["usd_remaining"] <= 0:
            __transition_to__("completed", "cost budget exhausted")
            return complete_result(state, "completed", "Cost budget exhausted.")

        # 3. Inject prior knowledge and activate skills on first step
        if step == 0:
            docs = __retrieve_docs__(goal, 5)
            if docs:
                knowledge = format_docs(docs)
                append_system_append(working_messages, knowledge)

            # Select and inject skills based on goal keywords
            all_skills = __list_skills__()
            active_skills = select_skills(all_skills, goal, max_candidates=3, max_tokens=6000)
            if active_skills:
                __set_active_skills__([
                    {
                        "doc_id": s.get("doc_id", ""),
                        "name": s.get("metadata", {}).get("name", "?"),
                        "version": s.get("metadata", {}).get("version", 1),
                        "snippet_names": [
                            sn.get("name", "")
                            for sn in s.get("metadata", {}).get("code_snippets", [])
                            if sn.get("name")
                        ],
                        "force_activated": False,
                    }
                    for s in active_skills
                ])
                skill_text = format_skills(active_skills)
                append_system_append(working_messages, skill_text)
                # Emit skill activation event for CLI/gateway display
                skill_names = ",".join(s.get("metadata", {}).get("name", "?") for s in active_skills)
                __emit_event__("skill_activated", skill_names=skill_names)
                # Store active skill IDs in state for tracking
                state["active_skill_ids"] = [s.get("doc_id", "") for s in active_skills]
                state["skill_snippet_names"] = []
                for s in active_skills:
                    for sn in s.get("metadata", {}).get("code_snippets", []):
                        state["skill_snippet_names"].append(sn.get("name", ""))

        # 3.5 Compact context before the next model call when needed.
        compact_if_needed(state, config)
        working_messages = ensure_working_messages(state, context)

        # 4. Call LLM
        __emit_event__("step_started", step=step)
        response = __llm_complete__(working_messages, actions, None)
        __emit_event__("step_completed", step=step,
                       input_tokens=response.get("usage", {}).get("input_tokens", 0),
                       output_tokens=response.get("usage", {}).get("output_tokens", 0))

        # 5. Handle response — code-only contract. Any other response type
        # (text / actions) means the adapter violated its contract: either
        # a backend returned something unexpected, a bridge adapter has a
        # bug, or a test mock is driving us through a deleted path. Fail
        # loudly so it's visible in logs and traces; silently treating it
        # as empty code would waste iterations until max_consecutive_errors.
        resp_type = response.get("type", "")
        if resp_type != "code":
            raise RuntimeError(
                "orchestrator expected response type 'code' under the "
                "code-only contract, got '" + str(resp_type) + "'. "
                "Some caller is still emitting text / action_calls — "
                "check the LlmBackend adapter or test mocks."
            )
        code = response.get("code", "")
        raw_content = response.get("content", "") or ""
        # Prefer the model's original content (which may include prose
        # around the code fence) so the transcript matches what it sent.
        assistant_text = raw_content if raw_content else "```repl\n" + code + "\n```"
        append_message(working_messages, "Assistant", assistant_text)

        # Execute code in nested Monty VM
        result = __execute_code_step__(code, state)

        # Update persisted state with results
        if result.get("return_value") is not None:
            state["step_" + str(step) + "_return"] = result["return_value"]
            state["last_return"] = result["return_value"]
        for r in result.get("action_results", []):
            state[r.get("action_name", "unknown")] = r.get("output")

        # Format output for next LLM context
        output = format_output(result)
        # Anchor: re-surface the goal at the tail so recency bias pulls the
        # model back to the original question instead of drifting toward
        # whatever the last tool output looked like. Skip step 0 — the goal
        # is already the most recent user message on the first turn.
        if step > 0:
            anchor = "ORIGINAL GOAL: " + str(goal)
            focus = state.get("current_focus")
            if focus:
                anchor += "\nCURRENT FOCUS: " + str(focus)
            anchor += (
                "\nYour response is a Python script — always. Not a "
                "conversational reply. If you've answered, the only valid "
                "script is `FINAL('<your answer>')`. Nothing else."
            )
            output = output + "\n\n---\n" + anchor
        append_message(working_messages, "User", output)

        # Check for FINAL() in code output
        if result.get("final_answer") is not None:
            __transition_to__("completed", "FINAL() in code")
            return complete_result(state, "completed", result["final_answer"])

        # Check for unified gate pause (new path)
        gate = result.get("pending_gate")
        if gate is None:
            gate = result.get("need_approval")
        if gate is not None and isinstance(gate, dict) and gate.get("gate_paused"):
            __save_checkpoint__(state, {
                "consecutive_errors": consecutive_errors,
                "compaction_count": state.get("compaction_count", 0),
            })
            __transition_to__("waiting", "gate paused: " + gate.get("gate_name", "unknown"))
            return {
                "outcome": "gate_paused",
                "state": state,
                "gate_name": gate.get("gate_name", ""),
                "action_name": gate.get("action_name", ""),
                "call_id": gate.get("call_id", ""),
                "parameters": gate.get("parameters", {}),
                "resume_kind": gate.get("resume_kind", {}),
            }

        # Check for approval or authentication needed (legacy path)
        if result.get("need_approval") is not None:
            approval = result["need_approval"]
            __save_checkpoint__(state, {
                "consecutive_errors": consecutive_errors,
                "compaction_count": state.get("compaction_count", 0),
            })
            if approval.get("need_authentication"):
                __transition_to__("waiting", "authentication needed")
                return {
                    "outcome": "need_authentication",
                    "state": state,
                    "credential_name": approval.get("credential_name", ""),
                    "action_name": approval.get("action_name", ""),
                    "call_id": approval.get("call_id", ""),
                    "parameters": approval.get("parameters", {}),
                }
            __transition_to__("waiting", "approval needed")
            return {
                "outcome": "need_approval",
                "state": state,
                "action_name": approval.get("action_name", ""),
                "call_id": approval.get("call_id", ""),
                "parameters": approval.get("parameters", {}),
            }

        # Track consecutive errors
        if result.get("had_error"):
            consecutive_errors += 1
            if max_consecutive_errors is not None and consecutive_errors >= max_consecutive_errors:
                __transition_to__("failed", "too many consecutive errors")
                return complete_result(
                    state,
                    "failed",
                    error=str(max_consecutive_errors) + " consecutive code errors",
                )
        else:
            consecutive_errors = 0

        __save_checkpoint__(state, {
            "consecutive_errors": consecutive_errors,
            "compaction_count": state.get("compaction_count", 0),
        })

    # Max iterations reached
    __transition_to__("completed", "max iterations reached")
    return complete_result(state, "max_iterations")


# Entry point: call run_loop with injected context variables
result = run_loop(context, goal, actions, state, config)
FINAL(result)

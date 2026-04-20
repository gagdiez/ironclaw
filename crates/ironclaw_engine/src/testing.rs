//! Test helpers exported so downstream crate tests can produce
//! well-formed `LlmResponse` fixtures without duplicating plumbing.
//!
//! Kept small on purpose — add only things that multiple test files need
//! and that must be consistent with the engine's code-only contract.

use crate::traits::llm::LlmOutput;
use crate::types::step::{LlmResponse, TokenUsage};

/// Wrap a plain text answer in a Python `FINAL(...)` call so a MockLlm
/// returning it will drive a CodeAct thread to completion under the
/// code-only contract. The answer is JSON-encoded so newlines and quotes
/// survive the round-trip through Monty.
///
/// Use this in test fixtures where the old contract would have returned
/// `LlmResponse::Text(msg)` directly — under the new contract that path
/// is rejected by the orchestrator.
pub fn code_final(msg: &str) -> LlmOutput {
    let encoded = serde_json::to_string(msg).expect("&str always encodes as JSON");
    let code = format!("FINAL({encoded})");
    LlmOutput {
        response: LlmResponse::Code {
            code: code.clone(),
            content: Some(format!("```repl\n{code}\n```")),
        },
        usage: TokenUsage::default(),
    }
}

//! Append-only JSONL dump of every LLM in/out and code-step in/out.
//! Hardcoded to /tmp/ironclaw-trace.jsonl. Tail it to see what's happening.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};

const TRACE_PATH: &str = "/tmp/ironclaw-trace.jsonl";

static SINK: OnceLock<Mutex<Option<File>>> = OnceLock::new();

fn sink() -> &'static Mutex<Option<File>> {
    SINK.get_or_init(|| {
        Mutex::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(TRACE_PATH)
                .ok(),
        )
    })
}

pub fn dump(kind: &str, thread_id: &str, payload: &serde_json::Value) {
    let line = serde_json::json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "thread_id": thread_id,
        "kind": kind,
        "payload": payload,
    });
    let Ok(text) = serde_json::to_string(&line) else { return };
    if let Ok(mut guard) = sink().lock()
        && let Some(file) = guard.as_mut()
    {
        let _ = writeln!(file, "{text}");
        let _ = file.flush();
    }
}

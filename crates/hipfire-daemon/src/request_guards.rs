//! Daemon-loop guards that answer or end the process before a request runs.

use std::io::Write;

/// A sticky GPU fault (HipError 700/719) leaves the HIP context dead; only a
/// new process gets a live one. The request that hit it has already been
/// answered, so exit 75 (EX_TEMPFAIL): `hipfire serve` respawns the daemon
/// and reloads the model instead of failing every later request.
pub(crate) fn exit_if_gpu_poisoned(stdout: &mut impl Write) {
    if let Some(poison) = hipfire_runtime::reset_core::gpu_poison() {
        eprintln!(
            "[daemon] GPU context dead after sticky HipError({}) at {}; exiting (75) for a process restart",
            poison.code, poison.site,
        );
        let _ = stdout.flush();
        std::process::exit(75);
    }
}

/// Why a tensor-parallel (EP) request cannot be served: EP decode has no
/// penalty sampler and the EP load carries no vision tower, so an image or an
/// explicit non-neutral penalty is refused instead of silently dropped.
fn ep_refusal(msg: &serde_json::Value, has_image: bool) -> Option<String> {
    if has_image {
        return Some(
            "images are not supported at tp>1 (the tensor-parallel load has no vision tower)"
                .to_owned(),
        );
    }
    [
        ("repeat_penalty", 1.0),
        ("repetition_penalty", 1.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
    ]
    .into_iter()
    .find(|(key, neutral)| {
        msg.get(*key)
            .and_then(|v| v.as_f64())
            .is_some_and(|value| value != *neutral)
    })
    .map(|(key, _)| {
        format!(
            "{key} is not supported at tp>1 (tensor-parallel decode has no penalty sampler); omit it or serve at tp=1"
        )
    })
}

/// At tp>1, answer a request [`ep_refusal`] rejects with an `unsupported`
/// error. Returns true when the request was refused and must be skipped.
pub(crate) fn refuse_ep_request(
    stdout: &mut impl Write,
    id: &str,
    msg: &serde_json::Value,
    ep: bool,
    has_image: bool,
) -> bool {
    let Some(message) = ep.then(|| ep_refusal(msg, has_image)).flatten() else {
        return false;
    };
    hipfire_generate::dense::emit_active_attempt_error(
        stdout,
        Some(id),
        &message,
        "unsupported",
        false,
        false,
    );
    let _ = stdout.flush();
    true
}

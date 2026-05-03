/// Dashboard roles for the elderly monitoring product (roadmap).
///
/// **Caretaker** — monitors up to two elders simultaneously: live activity, fall
/// probability, fall-type prediction; receives escalations when the elder does not
/// confirm after alarms.
///
/// **Elder** — receives fall confirmation dialogs; responses are logged for model QA
/// and improvement (false alarm, wrong type, need help, etc.).
///
/// **Admin** — operational view: all caretakers/elders, latency, accuracy proxies from
/// logged feedback, drift indicators.
///
/// System design details: repository `docs/ARCHITECTURE.md`.
/// Inference contract: `models/inference_manifest.json` + `scripts/inference/motion_pipeline.py`.

enum AppRole {
  caretaker,
  elder,
  admin,
}

/// Example responses after a fall alert (wire to backend `/api/v1/events/fall-feedback`).
enum FallFeedbackResponse {
  okay,
  needHelp,
  falseAlarm,
  wrongFallType,
  correctFallType,
  noHelpNeeded,
}

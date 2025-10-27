# GestureSpec JSON

Gesture specs live under `gestures/` and are the canonical description of a hand pose. They are typically created either by hand or via `scripts/gesture_to_train.py`, which can translate a natural-language description into a JSON object compliant with Mujosign's schema.

## Minimum Contract (currently consumed by the code)

```json
{
  "gesture": "ok_sign_2",
  "joints": {
    "thumb_MCP_flex": { "target": 60, "tolerance": 10, "weight": 1.0 },
    "index_MCP_flex": { "target": 90, "tolerance": 10, "weight": 1.0 }
  },
  "notes": "Thumb and index finger form a closed ring; other fingers relaxed."
}
```

Only the `gesture` string and the `joints` map are required by the current runtime. `notes` is optional metadata that helps humans remember intent.

### Joint entries

- `target` and `tolerance` are in **degrees**.
- `weight` rescales the squared error for that DOF (default 1.0).
- Joint names must match keys in `src/mujosign/utils/joint_names.DOF_MAP`. Examples: `thumb_MCP_flex`, `index_PIP_flex`, `wrist_pitch`.

## Full Schema (forward-looking)

`schemas/gesture_spec.schema.json` allows for richer constructs such as `handedness`, `relations`, `contacts`, `orientation`, detailed `stability` windows, and per-term `weights`. These fields reflect the future scoring vision, but the implementation has not yet consumed them.

When authoring specs manually:

1. Stick to the minimal contract above if you want to ensure compatibility with today's code.
2. If you enable the additional fields, `scripts/validate_specs.py` can guarantee the JSON matches the schema, but downstream scoring will ignore the extra sections until the corresponding code paths are implemented.

## Authoring Guidelines

- Prefer tolerances (±10–20°) instead of expecting exact angles; the solver and RL policy need some slack.
- Keep weights relative—heavier weights for critical joints, lighter ones for gesture-supporting fingers.
- Use `notes` to capture qualitative intent or camera framing hints. These strings are surfaced in generated README files inside `library/<gesture>/runs/<hash>/`.
- Regenerate specs through the LLM helper when experimenting with new gestures; it already understands Mujosign DOF naming and produces valid JSON most of the time.

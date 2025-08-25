# Mujosign Runbook

## Step-by-step

1. **Author Specs**: Write gesture JSONs into `gestures/`.
2. **Validate Specs**: Run `scripts/validate_specs.py` against schema.
3. **Load Env**: Create MyoSuite `myoHandPoseFixed-v0` with render if needed.
4. **Scoring**: Implement `scoring.py` per SPEC_SCORING.md.
5. **Solve**:
   - Try **Fast Path** (IK → static muscle optimization).
   - If fail, run **Synergy Search** (low-D CMA-ES).
6. **Validate**: Apply acceptance gates.
7. **Render**: Produce thumbnail from fixed camera config.
8. **Archive**: Save artifacts into `library/<gesture>/`.
9. **Summarize**: Run `scripts/summarize_library.py` to build reports.

## Troubleshooting
- Blank viewer → check `MUJOCO_GL` backend and working directory.
- High ROM penalties → loosen tolerances, verify DOF mapping.
- Unstable hold → increase damping, shorten step size, or reduce hold steps.

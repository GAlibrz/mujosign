# Methodology

We describe the methodological framework underlying Mujosign, a system for discovering physiologically plausible muscle activation vectors that reproduce declarative hand gestures within Meta's MyoSuite MuJoCo environments. The approach integrates symbolic gesture specification, differentiable simulation access, deterministic scoring, and both optimisation- and reinforcement-learning-based controllers. The pipeline is fully instrumented to preserve provenance and guarantee the reproducibility of reported results.

## Gesture Specification

Each gesture is defined by a JSON document located in `gestures/`. A specification enumerates degrees of freedom (DOFs) corresponding to MyoSuite hand joints and assigns to each DOF a triplet \((\theta_i^\star, \tau_i, w_i)\) representing the target joint angle in degrees, an admissible tolerance, and a scalar weight. Optional `notes` capture qualitative intent for subsequent interpretation. While the JSON schema (`schemas/gesture_spec.schema.json`) anticipates richer constructs such as contact constraints and stability windows, the present study focuses on joint-target constraints, which form the minimal contract consumed by the runtime.

Gesture authoring proceeds either manually or via `scripts/gesture_to_train.py`, which leverages an OpenAI chat model to translate natural-language descriptions into schema-compliant JSON. The workflow enforces DOF naming consistency with `src/mujosign/utils/joint_names.DOF_MAP`, ensuring that learned policies operate on a stable ordering of actuated joints.

## Simulation Interface

All interaction with the simulator is mediated by the `EnvAdapter` (`src/mujosign/sim_adapter.py`). The adapter wraps Gymnasium instances of MyoSuite's `myoHandPoseFixed-v0` environment, exposing:

- Joint angles in degrees, keyed by Mujosign DOF names, obtained by mapping simulator joint indices to the specification vocabulary.
- Fingertip Cartesian positions, enabling future extensions to relation-based scoring.
- The number and ordering of actuators, derived from `assets/discovery/<env-id>/actuator_names.json`, which act as canonical muscle labels.

The adapter preserves simulator determinism by abstaining from stochastic augmentations and by providing direct access to MuJoCo's state via Gymnasium's API.

## Scoring Function

The current scoring module (`src/mujosign/scoring.py`) implements a deterministic pose error metric. Given the simulated joint angles \(\theta_i\) and a specification triplet \((\theta_i^\star, \tau_i, w_i)\), the per-joint penalty is computed as:
\[
e_i = w_i \cdot \max\left(0, \lvert \theta_i - \theta_i^\star \rvert - \tau_i \right)^2.
\]
The total score reported in this work is \(\sum_i e_i\). The function returns a dictionary that reserves fields for additional metrics (relation error, contact error, energy expenditure, etc.), thereby preserving API compatibility with future scoring extensions without affecting the present results.

## Static Optimisation (FastPath)

For deterministic search we employ the coordinate-descent optimiser defined in `src/mujosign/solvers/fastpath.py`. The algorithm iteratively refines an activation vector \(a \in [0, 1]^{n_u}\) (where \(n_u\) is the number of actuators) by sweeping through each coordinate and probing additive and subtractive perturbations of magnitude \(\delta\). At each candidate activation:

1. The adapter holds the action constant for a configurable number of simulation steps to allow the hand pose to settle.
2. Joint angles are scored using the pose error metric.
3. The candidate is accepted if it yields a lower total score.

Step sizes follow a schedule \(\{\delta_1, \dots, \delta_K\}\) that typically decreases geometrically, terminating early if no improvement is observed during a sweep. The optimiser is deterministic for a fixed schedule and seed, providing a reproducible baseline for each gesture.

## Reinforcement Learning (SAC)

To explore policies capable of satisfying gestures through sequential control, we formulate the problem as a Gymnasium-compatible environment (`src/mujosign/rl/pose_env.PoseActivationEnv`). Observations concatenate:

1. A feature encoding of the specification \([ \theta_i^\star/90, \tau_i/90, w_i ]\) across all DOFs.
2. The current hand configuration \(\theta_i/90\), normalised by \(90^\circ\) for numerical stability.

Actions are continuous muscle activations in \([0, 1]^{n_u}\). Each chosen action is held for \(h\) simulator steps, after which the reward is computed as:
\[
r_t = - \text{score}(\theta_t, \text{spec}) - \lambda_{\text{effort}} \lVert a_t \rVert_2^2 - \lambda_{\text{smooth}} \lVert a_t - a_{t-1} \rVert_2^2.
\]
Episodes terminate when the total score falls below a success threshold or when a maximum number of decision steps is reached.

We train Soft Actor–Critic (SAC) agents using `scripts/train_rl_sac.py`, which incorporates:

- Deterministic evaluation rollouts at fixed intervals to snapshot the best-performing activation.
- Periodic checkpointing and CSV-based logging of reward progress.
- Optional resume functionality for long-running experiments.

This RL pathway complements the static solver by autonomously discovering control sequences that may exploit temporal dynamics.

## Artefact Management

All optimisation and RL evaluations invoke `src/mujosign/artifacts.write_run_artifacts`, producing immutable run directories under `library/<gesture>/runs/<hash>/`. Each directory contains:

- `activation.json`: the ordered list of muscles and their activation values.
- `scores.json`: the scoring breakdown.
- `pose_summary.json`: joint angles and fingertip positions at evaluation time.
- `gesture_spec.json`: the exact specification used.
- `provenance.json`: environment identifiers, solver configuration, version tags, and SHA-256 digests of the spec and muscle order.
- `README.md`: a human-readable summary with metric highlights.
- `thumb.png` (optional): a rendered frame when available.

Symlinks track the latest, best-total-score, and best-pose runs, simplifying subsequent analysis. The provenance hash ensures that identical inputs map to identical run directories, preventing divergence between reported and reproducible results.

## Implementation Notes

- Experiments target the `myoHandPoseFixed-v0` environment, but the adapter generalises to other MyoSuite hand configurations provided actuator and joint discovery data are available.
- All experiments fix the MuJoCo rendering backend to `egl` for headless operation and bind to a GPU when available.
- LLM-assisted gesture generation requires `openai` and `python-dotenv`; these dependencies are optional and do not affect the reproducibility of the published results once a spec file has been materialised.

The combination of declarative specifications, deterministic scoring, and reproducible artefact logging provides a methodological foundation suitable for rigorous evaluation and comparison across gesture sets and solver families.

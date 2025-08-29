#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse, json
from pathlib import Path
import numpy as np
import gymnasium as gym

import myosuite  # noqa: F401
import mujoco    # noqa: F401

from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order
from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score as score_fn
from mujosign.artifacts import write_run_artifacts, ProvenanceInputs

def resolve_env_id(raw_id: str) -> str:
    from gymnasium.envs import registry
    keys = list(registry.keys())
    if raw_id in keys: return raw_id
    cand = [k for k in keys if k.startswith(raw_id + "-v")]
    if cand: return cand[0]
    g = raw_id + "-v0"
    if g in keys: return g
    raise gym.error.NameNotFound(
        f"Env id '{raw_id}' not found; myo envs: {[k for k in keys if 'myo' in k.lower()]}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--trace", required=True, help=".npz from ActionRecorder")
    ap.add_argument("--gesture", required=True, help="gesture name for library/<gesture>/runs/<hash>")
    ap.add_argument("--spec-json", required=True, help="gesture spec JSON used for scoring")
    ap.add_argument("--library-root", default="library")
    ap.add_argument("--t-index", type=int, default=None, help="choose a specific step; default = best by min score")
    ap.add_argument("--hold", type=int, default=None, help="physics steps to settle before scoring (default meta.hold or 20)")
    args = ap.parse_args()

    data = np.load(args.trace, allow_pickle=True)
    actions = data["actions"].astype(np.float32)
    scores  = data["scores"].astype(np.float32) if "scores" in data else None
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    hold = int(args.hold if args.hold is not None else meta.get("hold", 20))

    # choose frame
    if args.t_index is not None:
        t = int(args.t_index)
        if t < 0 or t >= len(actions):
            raise IndexError(f"t-index {t} out of range [0,{len(actions)-1}]")
    else:
        if scores is not None and np.isfinite(scores).any():
            t = int(np.nanargmin(scores))
        else:
            t = len(actions) - 1  # fallback: last

    a_best = np.clip(actions[t], 0.0, 1.0)

    env_id = resolve_env_id(args.env_id)
    env = gym.make(env_id)
    muscle_order = load_muscle_order(env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    # apply and settle before reading pose
    adapter.reset()
    adapter.step(a_best)  # sets ctrl and steps once
    for _ in range(max(1, hold - 1)):
        adapter.step(a_best)

    # compute pose and score
    angles_deg = adapter.get_joint_angles_deg()
    spec = json.loads(Path(args.spec_json).read_text())
    br = score_fn(angles_deg, spec)

    # compose artifacts
    activation_card = {
        "env_id": env_id,
        "muscles": muscle_order,
        "activations": a_best.tolist(),
        "range": [0.0, 1.0],
        "units": "fraction",
        "time_horizon": f"t={t} (from trajectory)",
    }
    versions = {
        "mujosign": "0.0.1",
        "myosuite": getattr(__import__("myosuite"), "__version__", "?"),
        "mujoco": getattr(__import__("mujoco"), "__version__", "?"),
    }
    prov = ProvenanceInputs(
        env_id=env_id,
        gesture_spec=spec,
        solver_config={"name": "trace_extract", "t_index": t, "hold": hold},
        versions=versions,
        muscle_order=muscle_order,
    )
    pose_summary = {
        "joint_angles_deg": angles_deg,
        "tip_positions_m": {k: v.tolist() for k, v in adapter.get_fingertip_positions().items()},
        "palm_normal_world": None,
    }

    run_dir = write_run_artifacts(
        library_root=Path(args.library_root),
        gesture=args.gesture,
        prov=prov,
        activation=activation_card,
        scores={**br, "accepted": False},
        pose_summary=pose_summary,
        thumb_png_path=None,  # you can re-render thumbnail later
    )
    print("[trace_to_run] wrote →", run_dir)
    env.close()

if __name__ == "__main__":
    main()
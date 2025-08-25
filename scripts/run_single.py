#!/usr/bin/env python3
import argparse, json
import numpy as np
import gymnasium as gym
import myosuite

from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order

from pathlib import Path
from mujosign.artifacts import write_run_artifacts, ProvenanceInputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--spec", required=True, help="Path to gesture JSON (e.g., gestures/v_sign.json)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)

    # Stable muscle order (for later saving); not used in this pose-only smoke
    muscle_order = load_muscle_order(args.env_id)

    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    # Load spec
    with open(args.spec, "r") as f:
        spec = json.load(f)

    # Reset and run a constant action (zeros)
    obs, info = env.reset()
    zero_action = np.zeros((adapter.nu,), dtype=float)
    for _ in range(args.steps):
        obs, rew, terminated, truncated, info = adapter.step(zero_action)
        if terminated or truncated:
            obs, info = adapter.reset()

    # Read current state & score (pose-only)
    angles_deg = adapter.get_joint_angles_deg()
    breakdown = score(angles_deg, spec)
    print("angles_deg:", adapter.get_joint_angles_deg())

    print(f"[{args.env_id}] {spec.get('gesture')} :: total={breakdown['total']:.3f}  pose={breakdown['pose_error']:.3f}")

    # Build activation card for the last action you tried (zeros here; replace with optimized)
    muscle_order = load_muscle_order(args.env_id)
    activation_vec = zero_action.tolist()  # or your optimized vector

    activation_card = {
        "env_id": args.env_id,
        "muscles": muscle_order,
        "activations": activation_vec,
        "range": [0.0, 1.0],
        "units": "fraction",
        "time_horizon": "static"
    }

    pose_summary = {
        "joint_angles_deg": angles_deg,
        "tip_positions_m": {k: v.tolist() for k, v in adapter.get_fingertip_positions().items()},
        "palm_normal_world": None,
    }

    versions = {
        "mujosign": "0.0.1",
        "myosuite": getattr(myosuite, "__version__", "?"),
        "mujoco": getattr(__import__("mujoco"), "__version__", "?"),
    }

    solver_cfg = {"name": "coord_descent_v0", "seed": 0, "passes": 0}  # update when you add an optimizer
    prov = ProvenanceInputs(
        env_id=args.env_id,
        gesture_spec=spec,
        solver_config=solver_cfg,
        versions=versions,
        muscle_order=muscle_order,
    )

    run_dir = write_run_artifacts(
        library_root=Path("library"),
        gesture=spec["gesture"],
        prov=prov,
        activation=activation_card,
        scores={**breakdown, "accepted": False},  # set True when you pass gates
        pose_summary=pose_summary,
        thumb_png_path=None,  # or Path("reports/last_frame.png")
    )

    print("Artifacts written to:", run_dir)
    env.close()

    

if __name__ == "__main__":
    main()
    
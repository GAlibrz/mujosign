#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import gymnasium as gym
import myosuite
import mujoco
import imageio

from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order
from mujosign.artifacts import write_run_artifacts, ProvenanceInputs
from mujosign.solvers import get_solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--spec", required=True, help="Path to gesture JSON (e.g., gestures/v_sign.json)")
    ap.add_argument("--steps", type=int, default=50)      # currently unused (solver controls steps)
    ap.add_argument("--render", action="store_true")      # ignored by many MyoSuite envs
    ap.add_argument("--opt", default="fastpath")
    ap.add_argument("--opt-config", default=None)
    args = ap.parse_args()

    # Create env (render_mode often ignored by MyoSuite; harmless)
    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)

    # Stable muscle order (for artifacts)
    muscle_order = load_muscle_order(args.env_id)

    # Adapter
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    # Load spec
    with open(args.spec, "r") as f:
        spec = json.load(f)

    # ---- Choose & run solver ----
    solver = get_solver(args.opt)
    opt_kwargs = {}
    if args.opt_config:
        p = Path(args.opt_config)
        if p.suffix.lower() in (".yml", ".yaml"):
            import yaml
            opt_kwargs = yaml.safe_load(p.read_text())
        else:
            opt_kwargs = json.loads(p.read_text())

    best_a, breakdown, angles_deg = solver(
        adapter=adapter,
        spec=spec,
        scoring_fn=score,
        **opt_kwargs,
    )
    activation_vec = best_a.tolist()

    print("angles_deg:", angles_deg)
    print(f"[{args.env_id}] {spec.get('gesture')} :: total={breakdown['total']:.3f}  pose={breakdown['pose_error']:.3f}")

    # ---- Prepare artifacts ----
    activation_card = {
        "env_id": args.env_id,
        "muscles": muscle_order,
        "activations": activation_vec,
        "range": [0.0, 1.0],
        "units": "fraction",
        "time_horizon": "static",
    }

    pose_summary = {
        "joint_angles_deg": angles_deg,
        "tip_positions_m": {k: v.tolist() for k, v in adapter.get_fingertip_positions().items()},
        "palm_normal_world": None,
    }

    versions = {
        "mujosign": "0.0.1",
        "myosuite": getattr(myosuite, "__version__", "?"),
        "mujoco": getattr(mujoco, "__version__", "?"),
    }

    solver_cfg = {"name": args.opt, "config": opt_kwargs}
    prov = ProvenanceInputs(
        env_id=args.env_id,
        gesture_spec=spec,
        solver_config=solver_cfg,
        versions=versions,
        muscle_order=muscle_order,
    )

    # ---- Thumbnail (best-effort) ----
    thumb_path = None

    # 1) Try Gym rgb_array
    try:
        # Recreate env with rgb_array mode (render_mode may be ignored on some MyoSuite envs)
        rgb_env = gym.make(args.env_id, render_mode="rgb_array")
        _obs, _ = rgb_env.reset()
        frame = rgb_env.render()  # HxWx3 uint8 or None
        if frame is not None:
            thumb_path = Path("reports/last_frame.png")
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(thumb_path, frame)
        rgb_env.close()
    except Exception as e:
        print(f"WARN: gym rgb_array render failed ({e})")

    # 2) Try dm_control Physics if not yet rendered
    if thumb_path is None:
        try:
            from dm_control.mujoco import engine as dm_engine
            # Some MyoSuite envs expose a Physics via env.unwrapped._physics or similar
            physics = getattr(env.unwrapped, "physics", None) or getattr(env.unwrapped, "_physics", None)
            if physics is None:
                # As a last resort, make a transient Physics from the current model/data
                # (API varies; on some versions you can do Physics.from_model)
                physics = None
            if physics is not None and hasattr(physics, "render"):
                rgb = physics.render(height=480, width=480, camera_id=0)
                thumb_path = Path("reports/last_frame.png")
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(thumb_path, rgb)
            else:
                print("WARN: dm_control Physics not available; skipping thumbnail.")
        except Exception as e:
            print(f"WARN: dm_control render failed ({e}); skipping thumbnail.")

    # ---- Write artifacts ----
    run_dir = write_run_artifacts(
        library_root=Path("library"),
        gesture=spec["gesture"],
        prov=prov,
        activation=activation_card,
        scores={**breakdown, "accepted": False},  # flip True when you add gates
        pose_summary=pose_summary,
        thumb_png_path=thumb_path if (thumb_path and thumb_path.exists()) else None,
    )

    print("Artifacts written to:", run_dir)
    env.close()


if __name__ == "__main__":
    main()
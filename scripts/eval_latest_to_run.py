#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

# mujosign / myosuite
from mujosign.rl.pose_env import PoseActivationEnv  # just to ensure deps present
from mujosign.sim_adapter import EnvAdapter
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order
from mujosign.scoring import score
from mujosign.artifacts import write_run_artifacts, ProvenanceInputs
import myosuite, mujoco  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/proj/ciptmp/ry14vuci/mujosign_ckpts/v_sign_sac/latest.zip")
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--spec",   default="gestures/v_sign.json")
    ap.add_argument("--library-root", default="library")
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=512)
    args = ap.parse_args()

    # Load spec + env/adapter
    spec = json.loads(Path(args.spec).read_text())
    env = gym.make(args.env_id)
    muscle_order = load_muscle_order(args.env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    # Build initial observation (spec features + current angles), matching training
    dof_order = list(DOF_MAP.keys())
    def feat(spec_obj):
        feats, joints = [], spec_obj.get("joints", {})
        for d in dof_order:
            r = joints.get(d, {})
            feats.extend([
                float(r.get("target", 0.0)) / 90.0,
                float(r.get("tolerance", 0.0)) / 90.0,
                float(r.get("weight", 1.0)),
            ])
        return np.asarray(feats, dtype=np.float32)

    spec_vec = feat(spec)
    adapter.reset()
    ang = adapter.get_joint_angles_deg()
    ang_vec = np.asarray([float(ang.get(d, 0.0)) for d in dof_order], dtype=np.float32) / 90.0
    ob = np.concatenate([spec_vec, ang_vec], axis=0).astype(np.float32)

    # Load policy
    model = SAC.load(args.ckpt)

    # Roll out + pick best by score
    a_best, best_total, best_br, best_ang = None, float("inf"), None, None
    for _ in range(max(1, args.horizon)):
        a, _ = model.predict(ob, deterministic=True)
        a = np.clip(a, 0.0, 1.0).astype(np.float32)

        # hold to settle
        for _ in range(args.hold):
            _, _, term, trunc, _ = adapter.step(a)
            if term or trunc:
                adapter.reset()

        ang = adapter.get_joint_angles_deg()
        br = score(ang, spec)
        total = float(br["total"])
        if total < best_total:
            best_total = total
            a_best = a.copy()
            best_br = br
            best_ang = ang

        # update obs
        ang_vec = np.asarray([float(ang.get(d, 0.0)) for d in dof_order], dtype=np.float32) / 90.0
        ob = np.concatenate([spec_vec, ang_vec], axis=0).astype(np.float32)

    # Write run artifacts
    activation_card = {
        "env_id": args.env_id,
        "muscles": muscle_order,
        "activations": a_best.tolist(),
        "range": [0.0, 1.0],
        "units": "fraction",
        "time_horizon": f"{args.horizon}x{args.hold}",
    }
    versions = {
        "mujosign": "0.0.1",
        "myosuite": getattr(myosuite, "__version__", "?"),
        "mujoco": getattr(mujoco, "__version__", "?"),
    }
    prov = ProvenanceInputs(
        env_id=args.env_id,
        gesture_spec=spec,
        solver_config={"name": "rl_sac", "hold": args.hold, "horizon": args.horizon},
        versions=versions,
        muscle_order=muscle_order,
    )
    pose_summary = {
        "joint_angles_deg": best_ang,
        "tip_positions_m": {k: v.tolist() for k, v in adapter.get_fingertip_positions().items()},
        "palm_normal_world": None,
    }
    run_dir = write_run_artifacts(
        library_root=Path(args.library_root),
        gesture=spec["gesture"],
        prov=prov,
        activation=activation_card,
        scores={**best_br, "accepted": False},
        pose_summary=pose_summary,
        thumb_png_path=None,
    )
    env.close()

    # Minimal output: absolute run dir + hash
    run_dir = run_dir.resolve()
    print(str(run_dir))
    print(run_dir.name)  # the hash only

if __name__ == "__main__":
    main()
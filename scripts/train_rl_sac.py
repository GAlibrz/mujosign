#!/usr/bin/env python3
"""
Balanced-verbosity SAC trainer for PoseActivationEnv.

Key flags:
  --mode {normal,quiet,verbose}     Console verbosity preset (default: normal)
  --print-interval N                Print a tiny status line every N steps
  --eval-interval N                 Run a quick eval every N steps and report score
  --csv PATH                        Optional: write step-wise eval summary CSV

Example:
  python scripts/train_rl_sac.py \
    --env-id myoHandPoseFixed-v0 \
    --specs gestures/v_sign.json \
    --total-steps 120000 \
    --hold 40 --max-steps 1 --success 500 \
    --normalize-angles \
    --logdir runs/rl_sac \
    --save checkpoints/rl_sac_latest \
    --print-interval 5000 \
    --eval-interval 20000 \
    --mode normal \
    --csv runs/rl_sac/progress.csv
"""
import argparse, json, csv, time
from pathlib import Path
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from mujosign.rl.pose_env import PoseActivationEnv
from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order
from mujosign.artifacts import write_run_artifacts, ProvenanceInputs

import myosuite  # noqa: F401
import mujoco    # noqa: F401


def load_specs(path_or_dir: str):
    p = Path(path_or_dir)
    if p.is_dir():
        return [json.loads(fp.read_text()) for fp in sorted(p.glob("*.json"))]
    return [json.loads(p.read_text())]


def make_vec_env(env_kwargs, seed: int):
    def _thunk():
        env = PoseActivationEnv(**env_kwargs, seed=seed)
        return Monitor(env)
    return DummyVecEnv([_thunk])


def evaluate_and_write(model, env_id, spec, hold, horizon, library_root: Path, echo=True):
    env = gym.make(env_id)
    muscle_order = load_muscle_order(env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    dof_order = list(DOF_MAP.keys())

    def featurize_spec(spec_obj):
        feats = []
        joints = spec_obj.get("joints", {})
        for d in dof_order:
            r = joints.get(d, {})
            feats.extend([float(r.get("target", 0.0)) / 90.0,
                          float(r.get("tolerance", 0.0)) / 90.0,
                          float(r.get("weight", 1.0))])
        return np.asarray(feats, dtype=np.float32)

    spec_vec = featurize_spec(spec)
    adapter.reset()
    ang = adapter.get_joint_angles_deg()
    ang_vec = np.asarray([float(ang.get(d, 0.0)) for d in dof_order], dtype=np.float32) / 90.0
    ob = np.concatenate([spec_vec, ang_vec], axis=0).astype(np.float32)

    a_best, best_total, best_br, best_ang = None, float("inf"), None, None
    for _ in range(max(1, horizon)):
        a, _ = model.predict(ob, deterministic=True)
        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        for _ in range(hold):
            _, _, term, trunc, _ = adapter.step(a)
            if term or trunc:
                adapter.reset()
        ang = adapter.get_joint_angles_deg()
        br = score(ang, spec)
        total = float(br["total"])
        if total < best_total:
            best_total, a_best, best_br, best_ang = total, a.copy(), br, ang
        ang_vec = np.asarray([float(ang.get(d, 0.0)) for d in dof_order], dtype=np.float32) / 90.0
        ob = np.concatenate([spec_vec, ang_vec], axis=0).astype(np.float32)

    activation_card = {
        "env_id": env_id,
        "muscles": muscle_order,
        "activations": a_best.tolist(),
        "range": [0.0, 1.0],
        "units": "fraction",
        "time_horizon": f"{horizon}x{hold}",
    }
    versions = {
        "mujosign": "0.0.1",
        "myosuite": getattr(myosuite, "__version__", "?"),
        "mujoco": getattr(mujoco, "__version__", "?"),
    }
    prov = ProvenanceInputs(
        env_id=env_id,
        gesture_spec=spec,
        solver_config={"name": "rl_sac", "hold": hold, "horizon": horizon},
        versions=versions,
        muscle_order=muscle_order,
    )
    pose_summary = {
        "joint_angles_deg": best_ang,
        "tip_positions_m": {k: v.tolist() for k, v in adapter.get_fingertip_positions().items()},
        "palm_normal_world": None,
    }
    run_dir = write_run_artifacts(
        library_root=library_root,
        gesture=spec["gesture"],
        prov=prov,
        activation=activation_card,
        scores={**best_br, "accepted": False},
        pose_summary=pose_summary,
        thumb_png_path=None,
    )
    if echo:
        print(f"[eval] {spec.get('gesture','?')}: total={best_total:.3f} → {run_dir}")
    env.close()
    return float(best_total), run_dir


class HeartbeatCallback(BaseCallback):
    """
    Tiny, low-noise logger: prints every `print_interval` steps.
    If an EvalCallback is provided, also prints last eval mean reward/score.
    Optionally appends to a CSV: step, wall_s, eval_mean, best_eval.
    """
    def __init__(self, print_interval: int, csv_path: str | None, eval_cb: EvalCallback | None, mode: str):
        super().__init__()
        self.print_interval = max(1, int(print_interval))
        self.csv_path = csv_path
        self.eval_cb = eval_cb
        self.mode = mode
        self._t0 = time.time()
        self._last_print = 0
        if self.csv_path:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            if not Path(self.csv_path).exists():
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["step", "wall_s", "eval_mean", "best_eval"])

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step - self._last_print >= self.print_interval:
            self._last_print = step
            wall = time.time() - self._t0
            eval_mean = getattr(self.eval_cb, "last_mean_reward", None) if self.eval_cb else None
            best_eval = getattr(self.eval_cb, "best_mean_reward", None) if self.eval_cb else None
            if self.mode != "quiet":
                msg = f"[{step:,}] {wall:6.1f}s"
                if eval_mean is not None:
                    msg += f" | eval_mean={eval_mean:.3f}"
                if best_eval is not None:
                    msg += f" (best={best_eval:.3f})"
                print(msg, flush=True)
            if self.csv_path:
                with open(self.csv_path, "a", newline="") as f:
                    w = csv.writer(f); w.writerow([step, f"{wall:.2f}",
                                                   f"{eval_mean:.6f}" if eval_mean is not None else "",
                                                   f"{best_eval:.6f}" if best_eval is not None else ""])
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--specs", default="gestures")
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--success", type=float, default=1000.0)
    ap.add_argument("--lam-effort", type=float, default=0.01)
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--normalize-angles", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", default="runs/rl_sac")
    ap.add_argument("--save", default="checkpoints/rl_sac_latest")
    ap.add_argument("--eval-hold", type=int, default=20)
    ap.add_argument("--eval-horizon", type=int, default=1)
    ap.add_argument("--library-root", default="library")

    # balanced verbosity
    ap.add_argument("--mode", choices=["quiet", "normal", "verbose"], default="normal")
    ap.add_argument("--print-interval", type=int, default=5000)
    ap.add_argument("--eval-interval", type=int, default=20000)
    ap.add_argument("--csv", default=None, help="Optional CSV log path")
    args = ap.parse_args()

    # Load specs
    specs = load_specs(args.specs)

    # Training env
    env_kwargs = dict(
        env_id=args.env_id,
        specs=specs,
        specs_path=None,
        hold=args.hold,
        max_steps=args.max_steps,
        success_threshold=args.success,
        lambda_effort=args.lam_effort,
        lambda_smooth=args.lam_smooth,
        normalize_angles=args.normalize_angles,
    )
    vec_env = make_vec_env(env_kwargs, seed=args.seed)

    # Eval env (small, fast)
    eval_env_kwargs = dict(env_kwargs)
    eval_env = make_vec_env(eval_env_kwargs, seed=args.seed + 123)

    # SB3 verbosity & TB based on mode
    sb3_verbose = 2 if args.mode == "verbose" else (0 if args.mode == "quiet" else 1)
    tb_dir = None if args.mode == "quiet" else args.logdir

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=sb3_verbose,
        tensorboard_log=tb_dir,
        learning_rate=3e-4,
        batch_size=512,
        gamma=0.95,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed,
    )

    # Eval callback (reports mean episode reward on eval_env)
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=None,      # keep disk quiet
        log_path=None,
        eval_freq=args.eval_interval,
        deterministic=True,
        render=False,
        verbose=0,
        n_eval_episodes=3,              # small & quick
    )

    # Heartbeat prints + optional CSV
    hb_cb = HeartbeatCallback(
        print_interval=args.print_interval,
        csv_path=args.csv,
        eval_cb=eval_cb,
        mode=args.mode,
    )

    if args.mode != "quiet":
        print(f"[start] steps={args.total_steps:,} hold={args.hold} max_steps={args.max_steps} "
              f"specs={len(specs)} mode={args.mode}")

    model.learn(total_timesteps=args.total_steps,
                progress_bar=(args.mode == "verbose"),
                callback=[eval_cb, hb_cb])

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save)
    if args.mode != "quiet":
        print(f"[done] checkpoint → {args.save}")

    # Final evaluation + artifact write for each spec (normal/verbose),
    # or just the first spec (quiet).
    library_root = Path(args.library_root)
    final_specs = specs if args.mode != "quiet" else [specs[0]]
    for spec in final_specs:
        evaluate_and_write(
            model=model,
            env_id=args.env_id,
            spec=spec,
            hold=args.eval_hold,
            horizon=args.eval_horizon,
            library_root=library_root,
            echo=(args.mode != "quiet"),
        )
    if args.mode == "quiet":
        print("[quiet] done.")


if __name__ == "__main__":
    main()
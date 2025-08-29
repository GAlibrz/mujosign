#!/usr/bin/env python3
"""
Train a SAC policy on PoseActivationEnv and (optionally) resume from a checkpoint.

Minimal-but-useful logs:
- Console: brief progress prints every eval.
- TensorBoard (optional): if you open it later.
- CSV: step, reward_mean, success_rate, best_score → <logdir>/progress.csv
- Checkpoints: <save>.zip (always latest) + <save>-stepXXXXXX.zip (every eval)

Usage (fresh run):
  python scripts/train_rl_sac.py \
    --env-id myoHandPoseFixed-v0 \
    --specs gestures/v_sign.json \
    --total-steps 300000 \
    --hold 20 --max-steps 8 \
    --success 1000.0 \
    --logdir /proj/ciptmp/$USER/mujosign_runs/v_sign_sac_full \
    --save   /proj/ciptmp/$USER/mujosign_ckpts/v_sign_sac_full/latest

Usage (resume automatically if --save.zip exists):
  python scripts/train_rl_sac.py ... --resume

Usage (resume from a specific file):
  python scripts/train_rl_sac.py ... --resume-from /path/to/ckpt.zip
"""

import argparse, csv, json
from pathlib import Path
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from mujosign.rl.pose_env import PoseActivationEnv
from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order
from mujosign.rl.action_recorder import ActionRecorder

import myosuite  # noqa: F401
import mujoco    # noqa: F401


# ---------- helpers ----------

def load_specs(path_or_dir: str):
    p = Path(path_or_dir)
    if p.is_dir():
        return [json.loads(fp.read_text()) for fp in sorted(p.glob("*.json"))]
    return [json.loads(p.read_text())]


def make_vec_env(env_kwargs, seed: int, logdir: Path, hold: int):
    def _thunk():
        env = PoseActivationEnv(**env_kwargs, seed=seed)
        # Wrap with ActionRecorder to save trajectories
        trace_dir = Path(logdir) / "traces"
        return ActionRecorder(env, out_dir=trace_dir, env_hold=hold)
    return DummyVecEnv([_thunk])


def featurize_spec(spec_obj):
    dof_order = list(DOF_MAP.keys())
    feats = []
    joints = spec_obj.get("joints", {})
    for d in dof_order:
        r = joints.get(d, {})
        feats.extend([
            float(r.get("target", 0.0)) / 90.0,
            float(r.get("tolerance", 0.0)) / 90.0,
            float(r.get("weight", 1.0)),
        ])
    return np.asarray(feats, dtype=np.float32)


def evaluate_and_write(model: SAC, env_id: str, spec: dict, hold: int, horizon: int, library_root: Path):
    """Deterministic rollout on one spec, write run artifacts."""
    env = gym.make(env_id)
    muscle_order = load_muscle_order(env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    dof_order = list(DOF_MAP.keys())
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

    # Prepare artifacts
    from mujosign.artifacts import write_run_artifacts, ProvenanceInputs
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
    env.close()
    return run_dir, best_total


# ---------- callbacks ----------

class EvalSaveCallback(BaseCallback):
    def __init__(self, env_id, specs, eval_hold, eval_horizon, save_path_base: Path,
                 log_csv_path: Path, library_root: Path, eval_every_steps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.env_id = env_id
        self.specs = specs
        self.eval_hold = eval_hold
        self.eval_horizon = eval_horizon
        self.save_base = save_path_base
        self.log_csv_path = log_csv_path
        self.library_root = library_root
        self.eval_every = eval_every_steps

        self.best_total_seen = float("inf")
        self._csv_started = False
        log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_callback(self) -> None:
        # header
        if not self.log_csv_path.exists():
            with self.log_csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "reward_mean", "success_rate", "best_score"])
        self._csv_started = True

    def _on_step(self) -> bool:
        # run infrequently
        if self.n_calls % self.eval_every != 0:
            return True

        # quick eval: mean reward over 5 episodes + success rate
        ep_rewards = []
        successes = 0
        for _ in range(5):
            obs = self.training_env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            done, trunc = False, False
            total_r = 0.0
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=False)
                obs, r, done, trunc, info = self.training_env.step(action)
                total_r += float(r)
            ep_rewards.append(total_r)
            if "final_info" in info and info["final_info"]:
                finfo = info["final_info"]
                if isinstance(finfo, dict):
                    sc = finfo.get("score", {})
                    if isinstance(sc, dict) and float(sc.get("total", 1e9)) <= 1000.0:
                        successes += 1

        reward_mean = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        # write artifacts for each spec and checkpoint
        worst_total = 0.0
        for spec in self.specs:
            _, best_total = evaluate_and_write(
                model=self.model,
                env_id=self.env_id,
                spec=spec,
                hold=self.eval_hold,
                horizon=self.eval_horizon,
                library_root=self.library_root,
            )
            worst_total = max(worst_total, best_total)
            self.best_total_seen = min(self.best_total_seen, best_total)

        # save checkpoints (rolling + stepped)
        self.model.save(str(self.save_base))
        self.model.save(str(self.save_base.parent / f"{self.save_base.name}-step{self.num_timesteps:06d}"))

        # append CSV + concise print
        with self.log_csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([self.num_timesteps, reward_mean, successes / 5.0, self.best_total_seen])

        print(f"[eval] step={self.num_timesteps} reward_mean={reward_mean:.2f} "
              f"success_rate={successes/5.0:.2f} best_score={self.best_total_seen:.2f}", flush=True)

        return True


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--specs", default="gestures", help="JSON file or directory of JSON specs")
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--hold", type=int, default=20, help="sim steps per RL action")
    ap.add_argument("--max-steps", type=int, default=8, help="RL steps per episode")
    ap.add_argument("--success", type=float, default=1000.0)
    ap.add_argument("--lam-effort", type=float, default=0.01)
    ap.add_argument("--lam-smooth", type=float, default=0.0)
    ap.add_argument("--normalize-angles", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", default="runs/rl_sac")
    ap.add_argument("--save", default="checkpoints/rl_sac_latest")
    ap.add_argument("--eval-hold", type=int, default=20, help="hold when extracting final activation")
    ap.add_argument("--eval-horizon", type=int, default=1, help="policy steps when extracting final activation")
    ap.add_argument("--library-root", default="library")
    # resume options
    ap.add_argument("--resume", action="store_true", help="resume if --save.zip exists")
    ap.add_argument("--resume-from", default="", help="explicit checkpoint .zip to resume from")
    # eval/ckpt cadence (in environment steps)
    ap.add_argument("--eval-interval", type=int, default=10_000)
    args = ap.parse_args()

    # specs
    specs = load_specs(args.specs)

    # env
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
    vec_env = make_vec_env(env_kwargs, seed=args.seed, logdir=Path(args.logdir), hold=args.hold)
    save_base = Path(args.save)
    save_base.parent.mkdir(parents=True, exist_ok=True)

    # model (fresh vs resume)
    model = None
    ckpt_to_load = None
    if args.resume_from:
        ckpt_to_load = Path(args.resume_from)
    elif args.resume and save_base.with_suffix(".zip").exists():
        ckpt_to_load = save_base.with_suffix(".zip")

    if ckpt_to_load and ckpt_to_load.exists():
        print(f"[resume] loading checkpoint: {ckpt_to_load}")
        model = SAC.load(str(ckpt_to_load), env=vec_env, tensorboard_log=args.logdir)
    else:
        print("[start] new SAC run")
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=0,                     # keep console quiet; we print eval summaries
            tensorboard_log=args.logdir,
            learning_rate=3e-4,
            batch_size=512,
            gamma=0.95,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=args.seed,
        )

    # callback: periodic eval + save + CSV
    log_csv = Path(args.logdir) / "progress.csv"
    cb = EvalSaveCallback(
        env_id=args.env_id,
        specs=specs,
        eval_hold=args.eval_hold,
        eval_horizon=args.eval_horizon,
        save_path_base=save_base,
        log_csv_path=log_csv,
        library_root=Path(args.library_root),
        eval_every_steps=max(1, int(args.eval_interval)),
    )

    # train with progress bar
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=True)

    # final save
    model.save(str(save_base))
    print(f"[done] saved → {save_base.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
# SPDX-License-Identifier: MIT
import os, json, time
from pathlib import Path
import numpy as np
import gymnasium as gym

class ActionRecorder(gym.Wrapper):
    """
    Records per-step actions (and a few extras) for each episode and writes a .npz trace.
    Files land in out_dir/trace_<global_step>_<epoch>.npz (or timestamp if unknown).
    """
    def __init__(self, env, out_dir: str, env_hold: int = 1):
        super().__init__(env)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env_hold = int(env_hold)
        self._actions = []
        self._rewards = []
        self._scores = []
        self._angles = []
        self._t = 0
        self._episode_idx = 0
        self._global_step_getter = None  # optional: a callable returning global step (we set it from trainer)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # start a fresh episode buffer
        self._actions.clear()
        self._rewards.clear()
        self._scores.clear()
        self._angles.clear()
        self._t = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._rewards.append(float(reward))
        if isinstance(info, dict):
            # score breakdown total + angles if present
            sc = info.get("score", {})
            self._scores.append(float(sc.get("total", np.nan)))
            ang = info.get("angles_deg")
            # PoseActivationEnv doesn't currently put angles in info; we can compute from adapter:
            try:
                # best-effort: dig adapter to read angles (won't crash if not present)
                adapter = getattr(self.env.unwrapped, "_adapter", None)
                if adapter is not None:
                    ang_map = adapter.get_joint_angles_deg()
                    self._angles.append(np.asarray([ang_map.get(k, 0.0) for k in getattr(self.env.unwrapped, "_dof_order", [])], dtype=np.float32))
                else:
                    self._angles.append(np.nan)
            except Exception:
                self._angles.append(np.nan)
        else:
            self._scores.append(np.nan)
            self._angles.append(np.nan)

        self._t += 1
        if terminated or truncated:
            self._flush_trace(terminated=terminated, truncated=truncated)
            self._episode_idx += 1

        return obs, reward, terminated, truncated, info

    def _flush_trace(self, *, terminated: bool, truncated: bool):
        actions = np.stack(self._actions, axis=0) if len(self._actions) else np.zeros((0, self.env.action_space.shape[0]), dtype=np.float32)
        rewards = np.asarray(self._rewards, dtype=np.float32)
        scores  = np.asarray(self._scores, dtype=np.float32)
        angles  = np.asarray(self._angles, dtype=np.float32) if len(self._angles) and not np.isscalar(self._angles[0]) else None

        # name with global step if trainer provides it
        gstep = None
        if callable(self._global_step_getter):
            try:
                gstep = int(self._global_step_getter())
            except Exception:
                gstep = None
        stamp = f"gs{gstep}" if gstep is not None else time.strftime("%Y%m%d-%H%M%S")

        meta = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "episode_idx": int(self._episode_idx),
            "hold": int(self.env_hold),
        }

        out = self.out_dir / f"trace_{stamp}_ep{self._episode_idx:05d}.npz"
        np.savez_compressed(
            out,
            actions=actions,
            rewards=rewards,
            scores=scores,
            angles=angles if angles is not None else np.zeros((0,), dtype=np.float32),
            meta=json.dumps(meta),
        )
        # tiny text index
        with open(self.out_dir / "index.txt", "a") as f:
            f.write(str(out) + "\n")
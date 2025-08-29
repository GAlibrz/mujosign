#!/usr/bin/env python3
"""
View a saved trace (.npz) of actions in a MyoSuite env using the dm_control viewer.
- Replays the sequence step-by-step (with configurable hold per action)
- Loops forever (or a fixed number of repeats)
- No Gym render() needed; uses dm_control.viewer directly

Usage:
  python scripts/view_trace_dmcontrol.py \
    --env-id myoHandPoseFixed-v0 \
    --trace /path/to/trace_....npz \
    --hold 20 \
    --warmup 200 \
    --repeat 0       # 0 = loop forever
"""

import argparse, json
from pathlib import Path
from typing import Optional
import numpy as np

import gymnasium as gym
import myosuite  # registers myo* envs

# dm_env interface and viewer
import dm_env
from dm_env import specs
from dm_control import viewer as dc_viewer


def resolve_env_id(raw_id: str) -> str:
    """Pick a valid Gym env id even if the exact suffix isn't provided."""
    from gymnasium.envs import registry
    keys = list(registry.keys())
    if raw_id in keys:
        return raw_id
    suffixed = [k for k in keys if k.startswith(raw_id + "-v")]
    if suffixed:
        print(f"[view_trace] using '{suffixed[0]}'")
        return suffixed[0]
    loose = [k for k in keys if raw_id in k]
    if loose:
        print(f"[view_trace] using '{loose[0]}'")
        return loose[0]
    guess = raw_id + "-v0"
    if guess in keys:
        print(f"[view_trace] using '{guess}'")
        return guess
    raise gym.error.NameNotFound(
        f"Could not resolve env id '{raw_id}'. Registered myo envs: "
        f"{[k for k in keys if 'myo' in k.lower()]}"
    )


def find_dm_physics(env) -> Optional[object]:
    """
    Best-effort to find a dm_control Physics object (with .render()/.step()).
    We search through wrappers/attributes to locate it.
    """
    seen, stack = set(), [getattr(env, "unwrapped", env)]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        mod = getattr(o.__class__, "__module__", "")
        if hasattr(o, "render") and mod.startswith("dm_control"):
            return o
        for name in dir(o):
            if name.startswith("_"):
                continue
            try:
                child = getattr(o, name)
            except Exception:
                continue
            cm = getattr(child.__class__, "__module__", "")
            if any(k in cm for k in ("dm_control", "myosuite", "mujoco")) or hasattr(child, "render"):
                stack.append(child)
    return None


def load_actions(npz_path: Path) -> np.ndarray:
    """Load actions array from a trace .npz (keys: actions/action/acts)."""
    z = np.load(str(npz_path), allow_pickle=True)
    for k in ("actions", "action", "acts"):
        if k in z:
            arr = np.asarray(z[k], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"{k} has shape {arr.shape}, expected (T, A)")
            return arr
    raise RuntimeError(f"No action array in {npz_path}. Keys={list(z.keys())}")


class SeqActivationEnv(dm_env.Environment):
    """
    Minimal dm_env wrapper that feeds a SEQUENCE of activations to dm_control Physics.
    Each viewer step:
      - applies actions[t] to ctrl
      - runs `hold` physics steps
      - advances t, and loops when reaching the end
    """
    def __init__(self, physics, actions: np.ndarray, hold: int = 1, warmup: int = 200, repeat: int = 0):
        self._physics = physics
        self._acts = np.clip(actions.astype(np.float32), 0.0, 1.0)
        self._T, self._A = self._acts.shape
        self._hold = max(1, int(hold))
        self._warmup = max(0, int(warmup))
        self._repeat = repeat  # 0=loop forever, else number of passes to play
        self._nu = int(self._physics.model.nu)
        if self._A != self._nu:
            raise ValueError(f"Action dim {self._A} != model.nu {self._nu}")
        self._t = 0
        self._loop = 0

    @property
    def physics(self):
        return self._physics

    def reset(self) -> dm_env.TimeStep:
        # Reset physics, warm up a little to stabilize
        if hasattr(self._physics, "reset"):
            self._physics.reset()
        for _ in range(self._warmup):
            self._physics.step()
        self._t = 0
        return dm_env.restart(observation={})

    def step(self, action) -> dm_env.TimeStep:
        # Ignore external action; drive from recorded sequence
        a = self._acts[self._t]
        self._physics.data.ctrl[:self._nu] = a
        for _ in range(self._hold):
            self._physics.step()

        self._t += 1
        if self._t >= self._T:
            self._t = 0
            self._loop += 1
            # If repeat>0 and we've played that many loops, signal termination so viewer can stop
            if self._repeat > 0 and self._loop >= self._repeat:
                return dm_env.termination(reward=0.0, observation={})

        return dm_env.transition(reward=0.0, observation={})

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self._nu,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name="activation",
        )

    def observation_spec(self):
        return {}

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--trace", required=True, help=".npz file with action sequence (T,A)")
    ap.add_argument("--hold", type=int, default=20, help="physics steps per viewer step (per action)")
    ap.add_argument("--warmup", type=int, default=200, help="physics steps before first frame")
    ap.add_argument("--repeat", type=int, default=0, help="0 = loop forever; otherwise loop this many times")
    args = ap.parse_args()

    trace_path = Path(args.trace)
    acts = load_actions(trace_path)
    print("[view_trace] actions:", acts.shape, "min=", float(np.min(acts)), "max=", float(np.max(acts)))

    env_id = resolve_env_id(args.env_id)
    gym_env = gym.make(env_id)

    physics = find_dm_physics(gym_env)
    if physics is None:
        gym_env.close()
        raise RuntimeError("Couldn't find a dm_control Physics in this env; viewer not available via this wrapper.")

    seq_env = SeqActivationEnv(physics, acts, hold=args.hold, warmup=args.warmup, repeat=args.repeat)

    # A policy is required by dm_control.viewer, but we ignore it inside SeqActivationEnv.step
    def _policy(_time_step):
        # Return a dummy vector of correct shape; it's ignored anyway
        return np.zeros((int(physics.model.nu),), dtype=np.float32)

    try:
        dc_viewer.launch(environment_loader=lambda: seq_env, policy=_policy)
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse, json
from pathlib import Path
import numpy as np
import gymnasium as gym

import myosuite  # ensure MyoSuite envs are registered
import dm_env
from dm_env import specs

def resolve_env_id(raw_id: str) -> str:
    from gymnasium.envs import registry
    keys = list(registry.keys())
    if raw_id in keys: return raw_id
    cand = [k for k in keys if k.startswith(raw_id + "-v")]
    if cand: return cand[0]
    g = raw_id + "-v0"
    if g in keys: return g
    raise gym.error.NameNotFound(
        f"Env id '{raw_id}' not found; some myo envs: {[k for k in keys if 'myo' in k.lower()]}"
    )

def find_dm_physics(env):
    seen, stack = set(), [getattr(env, "unwrapped", env)]
    while stack:
        o = stack.pop()
        if id(o) in seen: continue
        seen.add(id(o))
        mod = getattr(o.__class__, "__module__", "")
        if hasattr(o, "render") and mod.startswith("dm_control"):
            return o
        for name in dir(o):
            if name.startswith("_"): continue
            try:
                child = getattr(o, name)
            except Exception:
                continue
            cm = getattr(child.__class__, "__module__", "")
            if any(k in cm for k in ("dm_control", "myosuite", "mujoco")) or hasattr(child, "render"):
                stack.append(child)
    return None

class TrajectoryEnv(dm_env.Environment):
    """
    Drives dm_control Physics with a pre-recorded activation sequence.
    Each viewer step advances 1 action, and we 'hold' it for H physics steps.
    """
    def __init__(self, physics, actions: np.ndarray, hold: int = 1, warmup: int = 200, loop: bool = True):
        self._physics = physics
        self._acts = actions.astype(np.float32)
        self._hold = max(1, int(hold))
        self._warmup = max(0, int(warmup))
        self._loop = bool(loop)
        self._t = 0
        self._nu = int(self._physics.model.nu)
        if self._acts.shape[1] != self._nu:
            raise ValueError(f"Actions shape {self._acts.shape} != (T, nu={self._nu})")

    @property
    def physics(self):
        return self._physics

    def reset(self):
        if hasattr(self._physics, "reset"):
            self._physics.reset()
        # Warm up on first action so pose is visible immediately
        if len(self._acts):
            self._physics.data.ctrl[:] = self._acts[0]
            for _ in range(self._warmup):
                self._physics.step()
        self._t = 0
        return dm_env.restart(observation={})

    def step(self, _action):
        if len(self._acts) == 0:
            return dm_env.transition(reward=0.0, observation={})

        a = self._acts[self._t]
        self._physics.data.ctrl[:] = a
        for _ in range(self._hold):
            self._physics.step()

        self._t += 1
        if self._t >= len(self._acts):
            if self._loop:
                self._t = 0
            else:
                return dm_env.termination(reward=0.0, observation={})
        return dm_env.transition(reward=0.0, observation={})

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(self._nu,), dtype=np.float32, minimum=0.0, maximum=1.0, name="activation")

    def observation_spec(self):
        return {}

    def close(self): pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--trace", required=True, help=".npz from ActionRecorder")
    ap.add_argument("--hold", type=int, default=None, help="override physics steps per frame (default: trace meta.hold)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    data = np.load(args.trace, allow_pickle=True)
    actions = data["actions"].astype(np.float32)
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    hold = int(args.hold if args.hold is not None else meta.get("hold", 1))

    env_id = resolve_env_id(args.env_id)
    gym_env = gym.make(env_id)
    physics = find_dm_physics(gym_env)
    if physics is None:
        gym_env.close()
        raise RuntimeError("No dm_control Physics found in env; viewer unavailable.")

    fixed = TrajectoryEnv(physics, actions, hold=hold, warmup=args.warmup, loop=args.loop)

    from dm_control import viewer as dc_viewer
    def _policy(_time_step):
        # Viewer requires a policy; we ignore it and feed the sequence inside env.step()
        return np.zeros((int(physics.model.nu),), dtype=np.float32)

    try:
        dc_viewer.launch(environment_loader=lambda: fixed, policy=_policy)
    finally:
        gym_env.close()

if __name__ == "__main__":
    main()
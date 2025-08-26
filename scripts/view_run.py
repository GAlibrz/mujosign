#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import gymnasium as gym

# Ensure MyoSuite registers envs
import myosuite  # noqa: F401

# dm_env interface for the viewer
import dm_env
from dm_env import specs


def resolve_env_id(raw_id: str) -> str:
    """Pick a valid Gym env id even if the suffix is missing."""
    from gymnasium.envs import registry
    keys = list(registry.keys())
    if raw_id in keys:
        return raw_id
    suffixed = [k for k in keys if k.startswith(raw_id + "-v")]
    if suffixed:
        print(f"[view_run] using '{suffixed[0]}'")
        return suffixed[0]
    loose = [k for k in keys if raw_id in k]
    if loose:
        print(f"[view_run] using '{loose[0]}'")
        return loose[0]
    guess = raw_id + "-v0"
    if guess in keys:
        print(f"[view_run] using '{guess}'")
        return guess
    raise gym.error.NameNotFound(
        f"Could not resolve env id '{raw_id}'. Registered myo envs: "
        f"{[k for k in keys if 'myo' in k.lower()]}"
    )


def find_dm_physics(env):
    """Best-effort to find a dm_control Physics object with .render() and .step()."""
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


class FixedActivationEnv(dm_env.Environment):
    """Minimal dm_env wrapper around a dm_control Physics to drive the viewer."""
    def __init__(self, physics, activation: np.ndarray, hold: int = 1, warmup: int = 200):
        self._physics = physics
        self._a = activation.astype(np.float32)
        self._hold = max(1, int(hold))
        self._warmup = max(0, int(warmup))
        self._t = 0
        self._nu = int(self._physics.model.nu)
        if self._a.shape[0] != self._nu:
            raise ValueError(f"Activation length {self._a.shape[0]} != model.nu {self._nu}")

    @property
    def physics(self):
        return self._physics

    def reset(self) -> dm_env.TimeStep:
        if hasattr(self._physics, "reset"):
            self._physics.reset()
        # Apply saved activations and warm up so pose is visible immediately.
        self._physics.data.ctrl[:] = self._a
        for _ in range(self._warmup):
            self._physics.step()
        self._t = 0
        return dm_env.restart(observation={})

    def step(self, action) -> dm_env.TimeStep:
        # Force holding the saved activations; ignore viewer-sent actions.
        self._physics.data.ctrl[:] = self._a
        for _ in range(self._hold):
            self._physics.step()
        self._t += 1
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
    ap.add_argument("--run-dir", required=True, help="library/<gesture>/runs/<hash>")
    ap.add_argument("--hold", type=int, default=1, help="physics steps per viewer step")
    ap.add_argument("--warmup", type=int, default=200, help="physics steps before opening the viewer")
    args = ap.parse_args()

    # Load saved activations
    run_dir = Path(args.run_dir)
    act = np.asarray(json.loads((run_dir / "activation.json").read_text())["activations"], dtype=float)

    # Print stats so you can see if activations are all ~0
    print("[view] act stats:",
          "len=", len(act),
          "min=", float(np.min(act)),
          "max=", float(np.max(act)),
          "mean=", float(np.mean(act)),
          "l2=", float(np.linalg.norm(act)))
    print("[view] first10:", np.array2string(act[:10], precision=3))

    # Resolve env id & make env
    env_id = resolve_env_id(args.env_id)
    gym_env = gym.make(env_id)

    # Get dm_control Physics
    physics = find_dm_physics(gym_env)
    if physics is None:
        gym_env.close()
        raise RuntimeError("Couldn't find a dm_control Physics in this env; viewer not available via this wrapper.")

    # Wrap in dm_env Environment and launch viewer; always return saved activations
    fixed_env = FixedActivationEnv(physics, act, hold=args.hold, warmup=args.warmup)

    from dm_control import viewer as dc_viewer
    def _policy(_time_step):
        return act  # force the saved activation every viewer step

    try:
        dc_viewer.launch(environment_loader=lambda: fixed_env, policy=_policy)
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
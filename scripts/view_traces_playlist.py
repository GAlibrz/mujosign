#!/usr/bin/env python3
"""
View a FOLDER of MyoSuite trace .npz files in sequence using the dm_control viewer.

- Collects *.npz under --traces-dir (optionally with --glob PATTERN)
- Each file is a sequence of actions (T, A) with key: actions / action / acts
- Plays each episode for --episode-repeats times (0 = endless loop per episode)
- Moves to the next file; loops through the folder (optionally --shuffle)
- Uses dm_control.viewer (no Gym render), so a window should open on desktop

Usage:
  python scripts/view_traces_playlist.py \
    --env-id myoHandPoseFixed-v0 \
    --traces-dir /path/to/traces \
    --hold 20 \
    --warmup 200 \
    --episode-repeats 1 \
    --shuffle 0

Press ESC / close window to exit.
"""

import argparse, random
from pathlib import Path
from typing import Optional, List
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
        print(f"[playlist] using '{suffixed[0]}'")
        return suffixed[0]
    loose = [k for k in keys if raw_id in k]
    if loose:
        print(f"[playlist] using '{loose[0]}'")
        return loose[0]
    guess = raw_id + "-v0"
    if guess in keys:
        print(f"[playlist] using '{guess}'")
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
            return np.clip(arr, 0.0, 1.0)
    raise RuntimeError(f"No action array in {npz_path}. Keys={list(z.keys())}")


class PlaylistEnv(dm_env.Environment):
    """
    dm_env wrapper that plays a playlist of action sequences on a dm_control Physics.

    - self._acts_list: List[np.ndarray (T, A)]
    - Plays current episode for 'episode_repeats' loops (or forever if 0), then advances.
    - Each viewer step: apply current actions[t], run 'hold' physics steps, advance t.
    """
    def __init__(
        self,
        physics,
        acts_list: List[np.ndarray],
        hold: int = 20,
        warmup: int = 200,
        episode_repeats: int = 1,  # 0 = endless per episode
    ):
        self._physics = physics
        self._acts_list = acts_list
        self._hold = max(1, int(hold))
        self._warmup = max(0, int(warmup))
        self._episode_repeats = int(episode_repeats)
        self._nu = int(self._physics.model.nu)

        # validate dims
        for arr in acts_list:
            if arr.shape[1] != self._nu:
                raise ValueError(f"Action dim {arr.shape[1]} != model.nu {self._nu} for one of the episodes.")

        self._epi_idx = 0
        self._t = 0
        self._loops_this_episode = 0

    @property
    def physics(self):
        return self._physics

    def _current_actions(self) -> np.ndarray:
        return self._acts_list[self._epi_idx]

    def _advance_episode(self):
        self._epi_idx = (self._epi_idx + 1) % len(self._acts_list)
        self._t = 0
        self._loops_this_episode = 0
        # Optional warm-up between episodes
        for _ in range(self._warmup):
            self._physics.step()

    def reset(self) -> dm_env.TimeStep:
        if hasattr(self._physics, "reset"):
            self._physics.reset()
        # initial warmup
        for _ in range(self._warmup):
            self._physics.step()
        self._t = 0
        self._loops_this_episode = 0
        print(f"[playlist] start episode {self._epi_idx+1}/{len(self._acts_list)} "
              f"({self._current_actions().shape[0]} steps)")
        return dm_env.restart(observation={})

    def step(self, action) -> dm_env.TimeStep:
        acts = self._current_actions()
        a = acts[self._t]
        self._physics.data.ctrl[:self._nu] = a
        for _ in range(self._hold):
            self._physics.step()

        self._t += 1
        if self._t >= acts.shape[0]:
            self._t = 0
            self._loops_this_episode += 1
            if self._episode_repeats > 0 and self._loops_this_episode >= self._episode_repeats:
                # advance episode automatically
                self._advance_episode()
                print(f"[playlist] next episode {self._epi_idx+1}/{len(self._acts_list)} "
                      f"({self._current_actions().shape[0]} steps)")
                # 👇 return restart instead of termination
                return dm_env.restart(observation={})

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
    ap.add_argument("--traces-dir", required=True, help="Directory containing .npz traces")
    ap.add_argument("--glob", default="trace_*.npz", help="Glob pattern within traces-dir")
    ap.add_argument("--hold", type=int, default=20, help="physics steps per action")
    ap.add_argument("--warmup", type=int, default=200, help="physics steps before first frame / between episodes")
    ap.add_argument("--episode-repeats", type=int, default=1, help="0 = loop same episode forever; else #loops before next")
    ap.add_argument("--shuffle", type=int, default=0, help="1 = shuffle the order once at start")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    paths = sorted(traces_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No traces found under {traces_dir} matching {args.glob}")

    # load all actions
    acts_list = []
    for p in paths:
        try:
            acts = load_actions(p)
            acts_list.append(acts)
        except Exception as e:
            print(f"[playlist] skip {p.name}: {e}")

    if not acts_list:
        raise RuntimeError("No valid traces found (none had an action array).")

    if args.shuffle:
        random.shuffle(acts_list)

    # Make env and find dm_control Physics
    env_id = resolve_env_id(args.env_id)
    gym_env = gym.make(env_id)
    physics = find_dm_physics(gym_env)
    if physics is None:
        gym_env.close()
        raise RuntimeError("Couldn't find a dm_control Physics in this env; viewer not available via this wrapper.")

    playlist_env = PlaylistEnv(
        physics=physics,
        acts_list=acts_list,
        hold=args.hold,
        warmup=args.warmup,
        episode_repeats=args.episode_repeats,
    )

    # Dummy policy; actions are ignored in step()
    def _policy(_ts):
        return np.zeros((int(physics.model.nu),), dtype=np.float32)

    try:
        dc_viewer.launch(environment_loader=lambda: playlist_env, policy=_policy)
    finally:
        gym_env.close()


if __name__ == "__main__":
    main()
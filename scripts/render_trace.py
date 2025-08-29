#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse, json
from pathlib import Path
import numpy as np
import gymnasium as gym

import myosuite  # noqa
import skvideo.io
from PIL import Image

def resolve_env_id(raw_id: str) -> str:
    from gymnasium.envs import registry
    keys = list(registry.keys())
    if raw_id in keys: return raw_id
    cand = [k for k in keys if k.startswith(raw_id + "-v")]
    if cand: return cand[0]
    g = raw_id + "-v0"
    if g in keys: return g
    raise gym.error.NameNotFound(f"Env id '{raw_id}' not found; got some myo envs: {[k for k in keys if 'myo' in k.lower()]}")

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--trace", required=True, help=".npz produced by ActionRecorder")
    ap.add_argument("--out", required=True, help="output .mp4 or directory for PNGs")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=int, default=1, help="physics steps per action frame (defaults to meta.hold if present)")
    args = ap.parse_args()

    data = np.load(args.trace, allow_pickle=True)
    actions = data["actions"].astype(np.float32)
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    hold = int(meta.get("hold", args.hold))

    env_id = resolve_env_id(args.env_id)
    env = gym.make(env_id)
    physics = find_dm_physics(env)
    if physics is None:
        raise RuntimeError("Could not find dm_control Physics in env for rendering")

    # writer: mp4 or PNG sequence
    out = Path(args.out)
    make_pngs = out.suffix.lower() not in (".mp4", ".mov", ".webm")
    if make_pngs:
        out.mkdir(parents=True, exist_ok=True)
        writer = None
    else:
        writer = skvideo.io.FFmpegWriter(
            str(out),
            inputdict={"-r": str(args.fps)},
            outputdict={"-vcodec":"libx264","-pix_fmt":"yuv420p","-r": str(args.fps)},
        )

    # reset and render
    if hasattr(physics, "reset"): physics.reset()

    for t, a in enumerate(actions):
        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        physics.data.ctrl[:] = a
        for _ in range(hold):
            physics.step()

        # render rgb
        frame = physics.render(height=args.height, width=args.width, camera_id=0)
        if make_pngs:
            Image.fromarray(frame).save(out / f"frame_{t:06d}.png")
        else:
            writer.writeFrame(frame)

    if writer is not None:
        writer.close()
    env.close()
    print("Wrote:", str(out))

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os, argparse, json, time
from pathlib import Path

import gymnasium as gym
import myosuite

def main():
    p = argparse.ArgumentParser(description="Sanity-run a MyoSuite env.")
    p.add_argument("--env-id", default="myoHandPoseFixed-v0")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--render", action="store_true")
    p.add_argument("--outdir", default="assets/discovery")
    p.add_argument("--gl", default=None, help="Override MUJOCO_GL (e.g., glfw, osmesa)")
    args = p.parse_args()

    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)

    outdir = Path(args.outdir) / args.env_id
    outdir.mkdir(parents=True, exist_ok=True)

    # Describe spaces
    info = {
        "env_id": args.env_id,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }
    with (outdir / "spaces.json").open("w") as f:
        json.dump(info, f, indent=2)

    # Reset + run a few random steps
    obs, info_reset = env.reset()
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)
        if terminated or truncated:
            obs, info_reset = env.reset()
        if args.render:
            time.sleep(0.01)
    env.close()

    print(f"✅ Sanity run OK — details saved to {outdir}/spaces.json")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse, time, numpy as np
from typing import Tuple
import gymnasium as gym
import myosuite
import mujoco
import numpy as np

from mujosign.sim_adapter import EnvAdapter
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order

def load_actions(path):
    z = np.load(path, allow_pickle=True)
    for k in ("actions","action","acts"):
        if k in z: return np.asarray(z[k], dtype=np.float32)
    raise RuntimeError(f"No action array in {path}. Keys={list(z.keys())}")

def get_dm_physics(env):
    # dm_control Physics is usually at env.unwrapped.sim
    un = getattr(env, "unwrapped", env)
    sim = getattr(un, "sim", None)
    if sim is None:
        raise RuntimeError("dm_control Physics not found on env.unwrapped.sim")
    return sim  # has .model and .data with .ptr

def wrap_mujoco_from_dm(sim) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    # Wrap existing dm_control pointers into mujoco python objects
    m = mujoco.MjModel.from_ptr(sim.model.ptr)
    d = mujoco.MjData.from_ptr(m, sim.data.ptr)
    return m, d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--sleep-ms", type=int, default=60)
    ap.add_argument("--sleep-mode", choices=["action","sim"], default="action")
    ap.add_argument("--repeat", type=int, default=0)  # 0 = loop forever
    args = ap.parse_args()

    actions = load_actions(args.trace)
    T, A = actions.shape
    print(f"Loaded {args.trace.split('/')[-1]}: steps={T}, act_dim={A}, hold={args.hold}")

    env = gym.make(args.env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES,
                         muscle_order=load_muscle_order(args.env_id))

    # Bridge dm_control → mujoco viewer
    sim = get_dm_physics(env)
    try:
        model, data = wrap_mujoco_from_dm(sim)
    except Exception as e:
        print(f"[warn] Could not wrap MuJoCo viewer ({e}). Falling back to headless stepping.")
        model = data = None

    viewer = None
    if model is not None and hasattr(mujoco, "viewer"):
        try:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(model, data)
            print("MuJoCo viewer opened. Close it or Ctrl+C to stop.")
        except Exception as e:
            print(f"[warn] Viewer launch failed: {e}")

    sleep_s = max(0, args.sleep_ms) / 1000.0
    repeats = args.repeat if args.repeat > 0 else float("inf")

    try:
        r = 0
        while r < repeats:
            r += 1
            adapter.reset()
            for t in range(T):
                a = np.clip(actions[t], 0.0, 1.0).astype(np.float32)
                for _ in range(max(1, args.hold)):
                    _, _, term, trunc, _ = adapter.step(a)
                    if viewer is not None:
                        viewer.sync()
                    if args.sleep_mode == "sim" and sleep_s:
                        time.sleep(sleep_s)
                    if term or trunc:
                        adapter.reset()
                if args.sleep_mode == "action" and sleep_s:
                    time.sleep(sleep_s)
            print(f"Replayed trace once (r={r}).")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            try: viewer.close()
            except: pass
        env.close()

if __name__ == "__main__":
    main()
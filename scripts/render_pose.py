#!/usr/bin/env python3
# scripts/render_pose.py
# Re-render a saved run: apply activation.json and (optionally) save a PNG.

import argparse, json
from pathlib import Path
from typing import Union, Optional

import numpy as np
import gymnasium as gym
import myosuite
import imageio

from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order


# --------- helpers ---------

def list_cameras_dm(model) -> list:
    """List camera names from a dm_control MjModel (names live in a bytes buffer)."""
    try:
        ncam = int(model.ncam)
        adr = np.array(model.name_camadr, dtype=int)
        buf = bytes(model.names)
        names = []
        for i in range(ncam):
            start = adr[i]
            end = buf.find(b"\x00", start)
            names.append(buf[start:end].decode("utf-8"))
        return names
    except Exception:
        return []


def find_dm_physics(env):
    """Best-effort BFS to find a dm_control Physics with .render()."""
    seen = set()
    stack = [getattr(env, "unwrapped", env)]
    while stack:
        o = stack.pop()
        oid = id(o)
        if oid in seen:
            continue
        seen.add(oid)
        mod = getattr(o.__class__, "__module__", "")
        if hasattr(o, "render") and mod.startswith("dm_control"):
            return o
        # Explore some likely children
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


def try_dm_render_with_camera(env, out_path: Path, req_h: int, req_w: int,
                              camera: Union[str, int, None]) -> bool:
    """Render via dm_control Physics, clamping to framebuffer, with optional upscale."""
    try:
        physics = find_dm_physics(env)
        if physics is None:
            return False

        # Framebuffer limits (offscreen)
        model = physics.model
        offw = int(getattr(model.vis.global_, "offwidth", 640))
        offh = int(getattr(model.vis.global_, "offheight", 480))

        # Clamp to framebuffer
        h = min(req_h, offh)
        w = min(req_w, offw)

        # Resolve camera id/name
        cam_id = camera if camera is not None else 0  # dm_control accepts int index or name (str)

        rgb = physics.render(height=h, width=w, camera_id=cam_id)  # HxWx3 uint8

        # Optional upscale to requested size for a nicer image
        if (h != req_h) or (w != req_w):
            try:
                from PIL import Image
                img = Image.fromarray(rgb)
                img = img.resize((req_w, req_h), Image.Resampling.LANCZOS)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)
                return True
            except Exception as e:
                # Fallback: write the original size
                print(f"WARN: upscale failed ({e}); writing framebuffer-sized image.")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(out_path, rgb)
                return True

        # Write as-is
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, rgb)
        return True

    except Exception as e:
        print(f"WARN: dm_control render failed ({e})")
        return False


def try_gym_rgb_array(env_id: str, action: np.ndarray, settle: int,
                      out_path: Path) -> bool:
    """Try Gym rgb_array; many MyoSuite envs ignore it, but attempt anyway."""
    try:
        rgb_env = gym.make(env_id, render_mode="rgb_array")
        obs, info = rgb_env.reset()
        for _ in range(settle):
            obs, rew, term, trunc, info = rgb_env.step(action)
            if term or trunc:
                obs, info = rgb_env.reset()
        frame = rgb_env.render()  # may be None
        rgb_env.close()
        if frame is None:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_path, frame)
        return True
    except Exception as e:
        print(f"WARN: gym rgb_array render failed ({e})")
        return False


# --------- main ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="myoHandPoseFixed-v0")
    ap.add_argument("--run-dir", required=True, help="Path to library/<gesture>/runs/<hash>/")
    ap.add_argument("--out", default="reports/replay.png", help="Output image path (PNG)")
    ap.add_argument("--settle", type=int, default=20, help="Sim steps to hold the action")
    ap.add_argument("--camera", default=None, help="dm_control camera name or index")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--list-cams", action="store_true", help="List available cameras and exit")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    act_path = run_dir / "activation.json"
    spec_path = run_dir / "gesture_spec.json"

    if not act_path.exists():
        raise FileNotFoundError(f"activation.json not found in {run_dir}")

    activation = json.loads(act_path.read_text())
    action = np.asarray(activation["activations"], dtype=float)

    # Build env + adapter
    env = gym.make(args.env_id)
    muscle_order = load_muscle_order(args.env_id)
    adapter = EnvAdapter(env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=muscle_order)

    # Optionally list camera names (dm_control)
    if args.list_cams:
        physics = find_dm_physics(env)
        cams = list_cameras_dm(physics.model) if physics is not None else []
        print("Available cameras:", cams if cams else "(none found)")
        env.close()
        return

    # Apply saved action for settle steps (to bring the hand into the saved pose basin)
    obs, info = adapter.reset()
    for _ in range(args.settle):
        obs, rew, term, trunc, info = adapter.step(action)
        if term or trunc:
            obs, info = adapter.reset()

    # Score if spec available
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        angles_deg = adapter.get_joint_angles_deg()
        breakdown = score(angles_deg, spec)
        print("angles_deg:", angles_deg)
        print(f"[replay] {args.env_id} {spec.get('gesture','?')} :: total={breakdown['total']:.3f}  pose={breakdown['pose_error']:.3f}")
    else:
        print("[replay] spec not found; skipping score.")

    # Try to render (prefer dm_control, then Gym)
    out_path = Path(args.out)
    ok = try_dm_render_with_camera(env, out_path, args.height, args.width, args.camera)
    if not ok:
        ok = try_gym_rgb_array(args.env_id, action, args.settle, out_path)

    if ok:
        print(f"[replay] wrote thumbnail → {out_path}")
    else:
        print("[replay] no thumbnail produced (renderer unavailable).")

    env.close()


if __name__ == "__main__":
    main()
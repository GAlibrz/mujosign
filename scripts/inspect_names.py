#!/usr/bin/env python3
import os, argparse, json, csv
from pathlib import Path

import numpy as np
import gymnasium as gym
import myosuite

# -------- name extraction via name_*adr + names buffer --------
def _decode_name_buf(buf, start: int) -> str:
    """
    Decode a C-string from model.names starting at byte offset `start`.
    Works for buf as bytes/bytearray or numpy uint8 array.
    """
    if start is None or start < 0:
        return ""

    # bytes / bytearray path
    if isinstance(buf, (bytes, bytearray)):
        n = len(buf)
        end = buf.find(b"\x00", start)
        if end == -1:
            end = n
        return buf[start:end].decode("utf-8", errors="ignore")

    # numpy array path
    arr = np.asarray(buf, dtype=np.uint8)
    n = int(arr.shape[0])
    end = start
    while end < n and arr[end] != 0:
        end += 1
    if end <= start:
        return ""
    return bytes(arr[start:end]).decode("utf-8", errors="ignore")


def _names_from_adr(model, adr_array_name: str, count: int):
    """
    Return list of names for an object type by reading model.name_*adr and model.names.
    """
    adr = getattr(model, adr_array_name, None)
    if adr is None:
        return []
    names_buf = getattr(model, "names", None)
    if names_buf is None:
        return []

    out = []
    for i in range(int(count)):
        start = int(adr[i])
        name = _decode_name_buf(names_buf, start)
        out.append(name if name else f"{adr_array_name[:-3]}_{i}")
    return out


def deg(x): return float(np.degrees(x))


def main():
    p = argparse.ArgumentParser(description="Inspect model names & ranges for a MyoSuite env.")
    p.add_argument("--env-id", default="myoHandPoseFixed-v0")
    p.add_argument("--outdir", default="assets/discovery")
    p.add_argument("--gl", default=None, help="Override MUJOCO_GL (e.g., glfw, osmesa)")
    args = p.parse_args()

    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl

    env = gym.make(args.env_id)
    sim = env.unwrapped.sim
    m, d = sim.model, sim.data

    outdir = Path(args.outdir) / args.env_id
    outdir.mkdir(parents=True, exist_ok=True)

    # --- summary ---
    summary = {
        "env_id": args.env_id,
        "njnt": int(getattr(m, "njnt", 0)),
        "nq": int(getattr(m, "nq", 0)),
        "nv": int(getattr(m, "nv", 0)),
        "nu": int(getattr(m, "nu", 0)),
        "nsite": int(getattr(m, "nsite", 0)),
        "ntendon": int(getattr(m, "ntendon", 0)),
        "nbody": int(getattr(m, "nbody", 0)),
        "ngeom": int(getattr(m, "ngeom", 0)),
    }
    with (outdir / "model_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- robust name extraction ---
    joints   = _names_from_adr(m, "name_jntadr",      summary["njnt"])
    sites    = _names_from_adr(m, "name_siteadr",     summary["nsite"])
    acts     = _names_from_adr(m, "name_actuatoradr", summary["nu"])
    tendons  = _names_from_adr(m, "name_tendonadr",   summary["ntendon"])
    bodies   = _names_from_adr(m, "name_bodyadr",     summary["nbody"])
    geoms    = _names_from_adr(m, "name_geomadr",     summary["ngeom"])

    for fname, vals in {
        "joint_names.json": joints,
        "site_names.json": sites,
        "actuator_names.json": acts,
        "tendon_names.json": tendons,
        "body_names.json": bodies,
        "geom_names.json": geoms,
    }.items():
        with (outdir / fname).open("w") as f:
            json.dump(vals, f, indent=2)

    # --- joint table (type, ranges, addrs) ---
    JT_HINGE, JT_SLIDE, JT_BALL, JT_FREE = 0, 1, 2, 3
    rows = []
    njnt = summary["njnt"]
    for j in range(njnt):
        name = joints[j]
        jtype = int(m.jnt_type[j])
        qposadr = int(m.jnt_qposadr[j])
        dofadr  = int(m.jnt_dofadr[j])
        limited = bool(m.jnt_limited[j])
        rng = np.array(m.jnt_range[j]) if getattr(m, "jnt_range", None) is not None else np.array([np.nan, np.nan])

        if jtype == JT_HINGE:
            r0, r1 = (deg(rng[0]), deg(rng[1])) if limited else (None, None)
            unit = "deg"
        elif jtype == JT_SLIDE:
            r0, r1 = (float(rng[0]), float(rng[1])) if limited else (None, None)
            unit = "m"
        else:
            r0, r1, unit = None, None, "n/a"

        rows.append({
            "joint_id": j,
            "name": name,
            "type": {JT_HINGE:"hinge", JT_SLIDE:"slide", JT_BALL:"ball", JT_FREE:"free"}.get(jtype, f"#{jtype}"),
            "qpos_addr": qposadr,
            "dof_addr": dofadr,
            "range_min": r0,
            "range_max": r1,
            "range_unit": unit,
            "limited": limited
        })

    if rows:
        with (outdir / "joint_table.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # --- actuator transmission (best-effort) ---
    act_rows = []
    nu = summary["nu"]
    trntype_arr = getattr(m, "actuator_trntype", None)
    trnid_arr   = getattr(m, "actuator_trnid", None)
    for a in range(nu):
        name = acts[a]
        trntype = int(trntype_arr[a]) if trntype_arr is not None else None
        trnid0  = int(trnid_arr[a,0]) if trnid_arr is not None else None
        target = None
        if trntype == 0 and trnid0 is not None:           # JOINT
            target = ("joint", joints[trnid0] if trnid0 < len(joints) else f"joint_{trnid0}")
        elif trntype == 2 and trnid0 is not None:         # TENDON
            target = ("tendon", tendons[trnid0] if trnid0 < len(tendons) else f"tendon_{trnid0}")

        act_rows.append({
            "actuator_id": a,
            "name": name,
            "trntype": trntype,
            "trnid0": trnid0,
            "target": target[0] if target else None,
            "target_name": target[1] if target else None,
        })

    if act_rows:
        with (outdir / "actuator_table.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(act_rows[0].keys()))
            w.writeheader()
            w.writerows(act_rows)

    env.close()
    print(f"✅ Model info saved under: {outdir}")

if __name__ == "__main__":
    main()
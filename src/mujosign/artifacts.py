# SPDX-License-Identifier: MIT
from __future__ import annotations
import hashlib, json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

# ---------- helpers ----------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _sha256_json(obj: Any) -> str:
    # Canonicalize with sorted keys and no whitespace
    s = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(s)

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _symlink_force(src: Path, dst: Path) -> None:
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        # On Windows or restricted FS, fall back to copying nothing (link optional)
        pass

# ---------- data ----------
@dataclass(frozen=True)
class ProvenanceInputs:
    env_id: str
    gesture_spec: Dict[str, Any]
    solver_config: Dict[str, Any]     # name, params, passes, seed
    versions: Dict[str, str]          # myosuite/mujoco/mujosign
    muscle_order: list[str]           # canonical names

    def hash(self) -> str:
        blob = {
            "env_id": self.env_id,
            "spec": self.gesture_spec,
            "solver": self.solver_config,
            "versions": self.versions,
            "muscles": self.muscle_order,
        }
        return _sha256_json(blob)[:16]  # short hash for folder name

# ---------- main API ----------
def write_run_artifacts(
    library_root: Path,
    gesture: str,
    prov: ProvenanceInputs,
    activation: Dict[str, Any],       # {env_id, muscles[], activations[], range, units, ...}
    scores: Dict[str, Any],           # scoring breakdown incl. total + accepted
    pose_summary: Dict[str, Any],     # joint angles, tips, palm normal
    thumb_png_path: Optional[Path] = None,
) -> Path:
    """
    Creates: library/<gesture>/runs/<hash>/{activation.json, scores.json, pose_summary.json,
             gesture_spec.json, provenance.json, thumb.png, README.md}
    Updates: library/<gesture>/index.json and symlinks latest/ & best_total/
    Returns: run directory Path
    """
    gesture_dir = library_root / gesture
    runs_dir = gesture_dir / "runs"
    run_id = prov.hash()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # hashes
    spec_sha = _sha256_json(prov.gesture_spec)
    muscles_sha = _sha256_json(prov.muscle_order)

    # write files
    _write_json(run_dir / "gesture_spec.json", prov.gesture_spec)
    _write_json(run_dir / "activation.json", activation)
    _write_json(run_dir / "scores.json", scores)
    _write_json(run_dir / "pose_summary.json", pose_summary)

    provenance = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env_id": prov.env_id,
        "versions": prov.versions,
        "solver": prov.solver_config,
        "spec_sha256": spec_sha,
        "muscle_order_sha256": muscles_sha,
        "run_id": run_id,
    }
    _write_json(run_dir / "provenance.json", provenance)

    # copy/rename thumbnail if provided
    if thumb_png_path and thumb_png_path.exists():
        (run_dir / "thumb.png").write_bytes(thumb_png_path.read_bytes())

    # README.md (per-run)
    accepted = bool(scores.get("accepted", False))
    readme = f"""# {gesture} — Run {run_id}

**Status:** {"ACCEPTED" if accepted else "REJECTED"}  
**Total Score:** {scores.get("total", "n/a")}  
**Date:** {provenance["timestamp"]}  
**Env:** {prov.env_id} | MyoSuite {prov.versions.get("myosuite","?")} | MuJoCo {prov.versions.get("mujoco","?")} | MujoSign {prov.versions.get("mujosign","?")}  
**Solver:** {prov.solver_config.get("name","?")} (seed={prov.solver_config.get("seed","?")})

![Thumbnail](thumb.png)

## 1) Summary
- Pose: {scores.get("pose_error")}
- Relation: {scores.get("relation_error")}
- Contact: {scores.get("contact_error")}
- Orientation: {scores.get("orientation_error")}
- Effort: {scores.get("effort")}
- Violations: ROM={scores.get("rom_violation")}, Tendon={scores.get("tendon_violation")}, Stability={scores.get("stability_violation")}

## 2) Achieved Pose
See `pose_summary.json`.

## 3) Activations
See `activation.json` (muscle names & values).

## 4) Repro
- Spec sha256: `{spec_sha}`
- Muscle order sha256: `{muscles_sha}`
- Full provenance: `provenance.json`
"""
    _write_text(run_dir / "README.md", readme)

    # Update gesture index.json and symlinks
    _update_gesture_index_and_links(gesture_dir, run_id, scores)

    return run_dir

def _update_gesture_index_and_links(gesture_dir: Path, run_id: str, scores: Dict[str, Any]) -> None:
    idx_path = gesture_dir / "index.json"
    idx = {"runs": [], "aliases": {}}
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text())
        except Exception:
            pass

    # append/replace this run (unique by id)
    idx["runs"] = [r for r in idx.get("runs", []) if r.get("id") != run_id]
    idx["runs"].append({
        "id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": float(scores.get("total", 0.0)),
        "pose_error": float(scores.get("pose_error", 0.0)),
        "accepted": bool(scores.get("accepted", False)),
        "solver": scores.get("solver", None),
    })
    # sort by time (optional) or by score; here keep insertion order

    _write_json(idx_path, idx)

    # update symlinks
    runs_dir = gesture_dir / "runs"
    this_run = runs_dir / run_id
    _symlink_force(this_run, gesture_dir / "latest")

    # best_total
    try:
        best = min(idx["runs"], key=lambda r: r.get("total", float("inf")))
        _symlink_force(runs_dir / best["id"], gesture_dir / "best_total")
    except ValueError:
        pass

    # best_pose
    try:
        bestp = min(idx["runs"], key=lambda r: r.get("pose_error", float("inf")))
        _symlink_force(runs_dir / bestp["id"], gesture_dir / "best_pose")
    except ValueError:
        pass
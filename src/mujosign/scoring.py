# src/mujosign/scoring.py
from __future__ import annotations
from typing import Dict, Any

def pose_error_deg(current_deg: Dict[str, float], spec: Dict[str, Any]) -> float:
    total = 0.0
    joints = spec.get("joints", {})
    for dof, rule in joints.items():
        target = float(rule.get("target", 0.0))
        tol    = float(rule.get("tolerance", 0.0))
        w      = float(rule.get("weight", 1.0))
        val    = float(current_deg.get(dof, 0.0))
        err    = abs(val - target) - tol
        if err > 0:
            total += w * (err * err)
    return total

def score(state_deg: Dict[str, float], spec: Dict[str, Any]) -> Dict[str, float]:
    pe = pose_error_deg(state_deg, spec)
    # Other terms to be added later
    return {
        "pose_error": pe,
        "relation_error": 0.0,
        "contact_error": 0.0,
        "orientation_error": 0.0,
        "effort": 0.0,
        "smoothness": 0.0,
        "rom_violation": 0.0,
        "tendon_violation": 0.0,
        "stability_violation": 0.0,
        "total": pe
    }
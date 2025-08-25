from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
# Mapping from our GestureSpec DOF names → actual model joint names (as in joint_names.json).
# This file is the single source of truth for naming.
#
# Notes:
# - Finger numbering in your model:
#     2 = index, 3 = middle, 4 = ring, 5 = little
# - 'pm#_flexion' ≈ PIP flexion, 'md#_flexion' ≈ DIP flexion.
# - Wrist: model exposes 'flexion' and 'deviation' (no explicit yaw/pitch),
#   we alias: wrist_pitch → flexion, wrist_yaw → deviation.

DOF_MAP = {
    # Index
    "index_MCP_flex": "mcp2_flexion",
    "index_PIP_flex": "pm2_flexion",   # proximal interphalangeal
    "index_DIP_flex": "md2_flexion",   # distal interphalangeal

    # Middle
    "middle_MCP_flex": "mcp3_flexion",
    "middle_PIP_flex": "pm3_flexion",
    "middle_DIP_flex": "md3_flexion",

    # Ring
    "ring_MCP_flex": "mcp4_flexion",
    "ring_PIP_flex": "pm4_flexion",
    "ring_DIP_flex": "md4_flexion",

    # Little
    "little_MCP_flex": "mcp5_flexion",
    "little_PIP_flex": "pm5_flexion",
    "little_DIP_flex": "md5_flexion",

    # Thumb
    "thumb_CMC_flex": "cmc_flexion",
    "thumb_MCP_flex": "mp_flexion",
    "thumb_IP_flex":  "ip_flexion",

    # Wrist aliases (spec uses pitch/yaw; model uses flexion/deviation)
    "wrist_pitch": "flexion",
    "wrist_yaw":   "deviation",
}

# Optional additional DOFs we might use later (uncomment if referenced by specs)
# DOF_MAP.update({
#     "thumb_CMC_abduction": "cmc_abduction",
#     "index_MCP_abduction": "mcp2_abduction",
#     "middle_MCP_abduction": "mcp3_abduction",
#     "ring_MCP_abduction": "mcp4_abduction",
#     "little_MCP_abduction": "mcp5_abduction",
# })

# Fingertip sites to use when computing relations (tip distances, etc.)
# Prefer actual tip sites (not the *_target ones).
TIP_SITES = {
    "thumb_tip":  "THtip",
    "index_tip":  "IFtip",
    "middle_tip": "MFtip",
    "ring_tip":   "RFtip",
    "little_tip": "LFtip",
}

# Palm site: the dump didn’t show a palm site. Leave None for now.
# If later find a site representing the palm frame, set PALM_SITE = "palm"
# and the orientation scorer can use it; otherwise compute a proxy normal.
PALM_SITE = None  # e.g., "palm" / "hand_root" if present in site_names.json

# Default location where scripts/inspect_names.py wrote names
_DISCOVERY_ROOT = Path("assets/discovery")

def load_muscle_order(env_id: str = "myoHandPoseFixed-v0") -> List[str]:
    """
    Preferred: read actuator_names.json produced by scripts/inspect_names.py.
    Fallback: raise a clear error so you remember to run discovery once.
    """
    path = _DISCOVERY_ROOT / env_id / "actuator_names.json"
    if not path.exists():
        raise FileNotFoundError(
            f"actuator_names.json not found at {path}. "
            "Run: python scripts/inspect_names.py --env-id {env_id}"
        )
    names = json.loads(path.read_text())
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid/empty actuator_names.json at {path}")
    return [str(n) for n in names]

# Keep a module-level alias if you want quick access:
# (Optional: set to None and always call load_muscle_order in code.)
MUSCLE_ORDER: Optional[List[str]] = None
#!/usr/bin/env python3
"""
gesture_to_train.py

Turn a natural-language gesture description into a mujosign gesture JSON
via OpenAI, save it under gestures/<name>.json, then invoke:

    scripts/launch_train_foreground.sh <name>

Requirements:
- OPENAI_API_KEY in env
- openai>=1.0.0 (new SDK: `pip install openai`)
- Your bash launcher at scripts/launch_train_foreground.sh

Example:
  python scripts/gesture_to_train.py \
    --name "ok_sign" \
    --prompt "Index finger and thumb touch to make a ring; other fingers upright; neutral wrist." \
    --model "gpt-4o-mini" \
    --temperature 0.2 \
    --dry-run    # (see JSON but do not launch training)

Notes:
- If you omit --name, the script asks the model to propose a short machine-safe name.
- If you pass --spec-file, it skips LLM and uses that JSON file verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# --- Optional: validate joint names against mujosign DOF list if present ---
_ALLOWED_DOFS = None
try:
    from mujosign.utils.joint_names import DOF_MAP as _DOF_MAP
    _ALLOWED_DOFS = set(_DOF_MAP.keys())
except Exception:
    _ALLOWED_DOFS = None  # validation will be looser if mujosign isn't importable

# --- OpenAI (new SDK) ---
try:
    from openai import OpenAI
except Exception as e:
    print("ERROR: The 'openai' package is required. Install with: pip install openai", file=sys.stderr)
    raise

from dotenv import load_dotenv
load_dotenv()  # this looks for .env in current or parent dirs


# ---------------- LLM prompting ----------------

SYSTEM_INSTRUCTIONS = """You are an assistant that produces strict JSON specs for a hand pose
gesture to be used by a MuJoCo/MyoSuite-based RL pipeline (mujosign).

Return ONLY a compact JSON object with the exact schema:

{
  "gesture": "<short_machine_name>",
  "joints": {
    "<dof_name>": { "target": <deg>, "tolerance": <deg>, "weight": <float> },
    ...
  },
  "notes": "<optional short human-readable hint>"
}

Rules:
- Degrees: flexion + is flex, - is extension. Wrist pitch/yaw in degrees.
- Tolerance: allowed absolute deviation around target, in degrees (e.g. 10–20).
- Weight: importance multiplier (1.0 default). Heavier weight for critical joints.
- Include only relevant DOFs. Omit joints not constrained by the description.
- Keep JSON minimal (no trailing commas, no comments, no code block fences).
"""

# If mujosign DOFs are known, tell the model to stick to them.
def build_user_prompt(prompt_text: str, allowed_dofs: set[str] | None, gesture_name_hint: str | None) -> str:
    dof_hint = ""
    if allowed_dofs:
        dof_list = ", ".join(sorted(allowed_dofs))
        dof_hint = f"\nValid DOF names (use only from this list): {dof_list}\n"
    name_hint = f"\nGesture short name hint: {gesture_name_hint}\n" if gesture_name_hint else ""
    return f"""Description:
{prompt_text}
{name_hint}{dof_hint}
Return ONLY the JSON object."""


def call_openai(prompt_text: str, model: str, temperature: float, gesture_name_hint: str | None) -> Dict[str, Any]:
    api_key=os.environ["OPENAI_API_KEY"]
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    user_prompt = build_user_prompt(prompt_text, _ALLOWED_DOFS, gesture_name_hint)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception as e:
        # As a fallback, try to extract the first JSON object substring
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Model did not return JSON. Raw content:\n{content}") from e
        data = json.loads(m.group(0))
    return data


# ---------------- Validation & IO ----------------

def validate_spec(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    if not isinstance(spec, dict):
        raise ValueError("Spec must be a JSON object.")
    if "gesture" not in spec or "joints" not in spec:
        raise ValueError("Spec must contain 'gesture' and 'joints' keys.")

    gesture = str(spec["gesture"]).strip()
    if not gesture:
        raise ValueError("'gesture' must be a non-empty string.")
    # sanitize to safe filename
    gesture_safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", gesture).lower()

    joints = spec["joints"]
    if not isinstance(joints, dict):
        raise ValueError("'joints' must be an object mapping dof -> rule.")

    for dof, rule in joints.items():
        if not isinstance(rule, dict):
            raise ValueError(f"Joint '{dof}' rule must be an object.")
        # optional dof name restriction
        if _ALLOWED_DOFS and dof not in _ALLOWED_DOFS:
            raise ValueError(f"Joint '{dof}' is not a valid DOF for this env.")
        for k in ("target", "tolerance", "weight"):
            if k not in rule:
                raise ValueError(f"Joint '{dof}' missing '{k}'.")
        try:
            float(rule["target"])
            float(rule["tolerance"])
            float(rule["weight"])
        except Exception:
            raise ValueError(f"Joint '{dof}' has non-numeric target/tolerance/weight.")
    return gesture_safe, joints


def write_spec(gesture_name: str, spec: Dict[str, Any], gestures_dir: Path) -> Path:
    gestures_dir.mkdir(parents=True, exist_ok=True)
    out_path = gestures_dir / f"{gesture_name}.json"
    with out_path.open("w") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
    return out_path


def launch_training(gesture_name: str, repo_root: Path, extra_args: list[str]) -> int:
    launcher = repo_root / "scripts" / "launch_train_foreground.sh"
    if not launcher.exists():
        raise FileNotFoundError(f"Launcher not found: {launcher}")
    cmd = ["bash", str(launcher), gesture_name] + extra_args
    print(f"\n[launch] {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="", help="Natural-language description of the gesture.")
    ap.add_argument("--name", default="", help="Optional short machine name. If empty, the model proposes one.")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--spec-file", default="", help="Skip LLM and use this JSON file instead.")
    ap.add_argument("--dry-run", action="store_true", help="Only write JSON; do not launch training.")
    ap.add_argument("--extra-train-arg", action="append", default=[],
                    help="Extra args to pass to launch_train_foreground.sh (repeatable).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    gestures_dir = repo_root / "gestures"

    # 1) Get (or load) the spec
    if args.spec_file:
        spec = json.loads(Path(args.spec_file).read_text())
    else:
        if not args.prompt:
            print("ERROR: Provide either --spec-file or --prompt.", file=sys.stderr)
            sys.exit(2)
        spec = call_openai(
            prompt_text=args.prompt,
            model=args.model,
            temperature=args.temperature,
            gesture_name_hint=(args.name or None),
        )

    # 2) Validate + reconcile gesture name
    gesture_safe, joints = validate_spec(spec)
    if args.name:
        # force the filename to user-provided name, but keep spec["gesture"] as-is
        gesture_file_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", args.name).lower()
    else:
        gesture_file_name = gesture_safe

    # 3) Write JSON
    out_path = write_spec(gesture_file_name, spec, gestures_dir)
    print(f"[ok] wrote gesture spec → {out_path}")

    # 4) Optionally launch training
    if args.dry_run:
        print("[dry-run] not launching training.")
        return

    rc = launch_training(gesture_file_name, repo_root, args.extra_train_arg)
    if rc != 0:
        print(f"[error] launcher exited with code {rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Validate gesture spec JSON files against schemas/gesture_spec.schema.json.

Usage:
  python scripts/validate_specs.py
  python scripts/validate_specs.py gestures/v_sign.json
  python scripts/validate_specs.py gestures/*.json
  python scripts/validate_specs.py path/to/dir
"""

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, exceptions as js_exc

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "gesture_spec.schema.json"
DEFAULT_TARGET = REPO_ROOT / "gestures"


def load_schema(schema_path: Path):
    try:
        with schema_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Schema not found: {schema_path}", file=sys.stderr)
        sys.exit(2)


def iter_json_files(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".json":
            yield path
        return
    if path.is_dir():
        yield from sorted(p for p in path.glob("*.json") if p.is_file())


def pretty_error(error: js_exc.ValidationError, file: Path) -> str:
    # Build a dotted path to the failing element
    loc = "".join(f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in error.path)
    loc = loc.lstrip(".")
    schema_loc = "/".join(str(p) for p in error.schema_path)
    return (
        f"  ↳ at `{loc or '<root>'}`: {error.message}\n"
        f"    schema path: {schema_loc}"
    )


def validate_file(validator: Draft202012Validator, file: Path) -> bool:
    try:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ {file} — invalid JSON: {e}", file=sys.stderr)
        return False

    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if not errors:
        print(f"✅ {file} — OK")
        return True

    print(f"❌ {file} — {len(errors)} error(s):")
    for e in errors:
        print(pretty_error(e, file), file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate gesture specs against JSON Schema.")
    parser.add_argument("paths", nargs="*", help="Files or directories to validate. Default: gestures/")
    args = parser.parse_args()

    schema = load_schema(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    targets = args.paths or [str(DEFAULT_TARGET)]
    any_fail = False

    for p in targets:
        path = Path(p).resolve()
        files = list(iter_json_files(path))
        if not files:
            print(f"(No JSON files found in {path})")
            continue
        for f in files:
            ok = validate_file(validator, f)
            any_fail = any_fail or (not ok)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
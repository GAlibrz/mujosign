#!/usr/bin/env python3
import json, re
from pathlib import Path

ROOT = Path("assets/discovery/myoHandPoseFixed-v0")

def load(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else []

def pick(candidates, *patterns):
    pats = [re.compile(p, re.I) for p in patterns]
    out = []
    for c in candidates:
        if any(p.search(c) for p in pats):
            out.append(c)
    return out

joints = load("joint_names.json")
sites  = load("site_names.json")
acts   = load("actuator_names.json")

print("# JOINTS — likely flexion DOFs")
for label, pats in {
    "index":  ("index|ind", "mcp|pip|dip|flex"),
    "middle": ("middle|mid", "mcp|pip|dip|flex"),
    "ring":   ("ring", "mcp|pip|dip|flex"),
    "little": ("little|pink|pinky", "mcp|pip|dip|flex"),
    "thumb":  ("thumb|th", "cmc|mcp|ip|flex"),
    "wrist":  ("wrist|wr", "pitch|yaw|flex|abd|dev"),
}.items():
    hits = pick(joints, *pats)
    print(f"{label:>6}:", hits)

print("\n# SITES — tips & palm")
for label, pats in {
    "tips": ("tip|distal",),
    "palm": ("palm|hand_root|hand_base",),
}.items():
    hits = pick(sites, *pats)
    print(f"{label:>6}:", hits)

print("\n# ACTUATORS (optional, first 20):")
print(acts[:20])
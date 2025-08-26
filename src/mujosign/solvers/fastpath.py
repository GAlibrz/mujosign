# src/mujosign/solvers/fastpath.py
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Callable, Dict, Any, Tuple, Optional
import numpy as np
import math
import random


def _rollout_static_action(
    adapter,
    action: np.ndarray,
    steps: int = 30,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply a constant action for `steps`, then read joint angles and score via the provided scorer.

    Returns:
        angles_deg: dict[str, float]  # current joint angles in degrees (per DOF_MAP keys)
        info: dict with any extra state you may want to augment later (reserved)
    """
    # Gymnasium API: reset -> (obs, info), step -> (obs, reward, terminated, truncated, info)
    obs, info0 = adapter.reset()
    for _ in range(steps):
        obs, rew, terminated, truncated, info = adapter.step(action)
        if terminated or truncated:
            obs, info0 = adapter.reset()
    angles_deg = adapter.get_joint_angles_deg()
    return angles_deg, {"obs": obs}


def optimize(
    adapter,
    spec: Dict[str, Any],
    scoring_fn: Callable[[Dict[str, float], Dict[str, Any]], Dict[str, float]],
    *,
    steps_per_eval: int = 30,
    passes: int = 3,
    step_schedule: Optional[list] = None,
    seed: int = 0,
    effort_weight: float = 0.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Coordinate-descent over activations a ∈ [0,1]^nu for a *static pose*.

    Strategy:
      - Start at zeros
      - For each pass (with decreasing step size δ):
          for i in [0..nu-1]:
             try a[i]+=δ  (clipped 0..1)  -> score_up
             try a[i]-=δ  (clipped 0..1)  -> score_dn
             keep the change that improves total score (pose_error-dominant)
      - Optional: add tiny effort regularization to bias towards smaller activations

    Args:
        adapter: EnvAdapter (must expose .nu, .reset(), .step(), .get_joint_angles_deg())
        spec: gesture spec dict (already validated)
        scoring_fn: function(state_angles_deg, spec) -> breakdown dict with 'total'/'pose_error' keys
        steps_per_eval: rollout steps for each candidate action
        passes: number of coordinate sweeps
        step_schedule: list of step sizes; default = [0.2, 0.1, 0.05][:passes]
        seed: RNG seed (kept for future randomized orders; fixed order by default)
        effort_weight: optional λ to add λ * sum(a^2) to the total
        verbose: print small progress logs

    Returns:
        best_action: np.ndarray shape (nu,)
        best_breakdown: dict[str, float] from scoring_fn
        best_angles_deg: dict[str, float]
    """
    rng = random.Random(seed)
    nu = getattr(adapter, "nu", None) or getattr(adapter.env.action_space, "shape", [None])[0]
    if not isinstance(nu, int):
        raise ValueError("Cannot determine action dimension (nu). Ensure adapter.nu is set.")

    if step_schedule is None:
        step_schedule = [0.2, 0.1, 0.05]
    if passes > len(step_schedule):
        # extend by halving last step if needed
        last = step_schedule[-1]
        step_schedule = step_schedule + [last * (0.5 ** k) for k in range(1, passes - len(step_schedule) + 1)]
    else:
        step_schedule = step_schedule[:passes]

    # Utility to evaluate a static action
    def eval_action(a: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        angles, _ = _rollout_static_action(adapter, a, steps=steps_per_eval)
        breakdown = scoring_fn(angles, spec)
        total = float(breakdown.get("total", math.inf))
        if effort_weight > 0:
            total = total + effort_weight * float(np.sum(a * a))
            # Reflect the regularized total in a copy (don’t mutate original scoring_fn output)
            breakdown = dict(breakdown)
            breakdown["total"] = total
            breakdown["effort"] = float(breakdown.get("effort", 0.0)) + float(np.sum(a * a)) * effort_weight
        return total, breakdown, angles

    # Start at zeros
    a = np.zeros((nu,), dtype=float)
    best_total, best_breakdown, best_angles = eval_action(a)

    if verbose:
        print(f"[fastpath] init total={best_total:.4f}")

    # Coordinate descent
    for pass_idx, delta in enumerate(step_schedule, 1):
        improved = False
        if verbose:
            print(f"[fastpath] pass {pass_idx}/{passes}  step={delta}")

        # Fixed order (0..nu-1); keep deterministic for reproducibility
        for i in range(nu):
            base_a = a.copy()

            # Try +delta
            cand_up = base_a.copy()
            cand_up[i] = min(1.0, max(0.0, cand_up[i] + delta))
            up_total, up_breakdown, up_angles = eval_action(cand_up)

            # Try -delta
            cand_dn = base_a.copy()
            cand_dn[i] = min(1.0, max(0.0, cand_dn[i] - delta))
            dn_total, dn_breakdown, dn_angles = eval_action(cand_dn)

            # Keep the best of {up, dn, base}
            if up_total < best_total and up_total <= dn_total:
                a = cand_up
                best_total, best_breakdown, best_angles = up_total, up_breakdown, up_angles
                improved = True
                if verbose:
                    print(f"[fastpath]  i={i:02d}  +{delta:.3f}  total={best_total:.4f}")
            elif dn_total < best_total and dn_total < up_total:
                a = cand_dn
                best_total, best_breakdown, best_angles = dn_total, dn_breakdown, dn_angles
                improved = True
                if verbose:
                    print(f"[fastpath]  i={i:02d}  -{delta:.3f}  total={best_total:.4f}")
            # else keep base; no change

        if verbose:
            print(f"[fastpath] end pass {pass_idx}: total={best_total:.4f}")

        # Early stop if no change in this pass
        if not improved:
            if verbose:
                print("[fastpath] no improvement; stopping early.")
            break

    return a, best_breakdown, best_angles
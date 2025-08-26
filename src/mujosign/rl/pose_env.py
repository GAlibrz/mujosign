# src/mujosign/rl/pose_env.py
# SPDX-License-Identifier: MIT
"""
PoseActivationEnv — Gymnasium environment for learning muscle activations
that satisfy a target hand gesture (pose) in MyoSuite/MuJoCo.

Observation:
    [ spec_features , current_joint_angles ]
    - spec_features: for each DOF in a stable order, [target, tolerance, weight]
    - current_joint_angles: ordered by the same DOF order (degrees or normalized)

Action:
    Continuous muscle activations in [0, 1]^nu (nu = number of actuators).

Reward:
    r = - total_score(spec, state) - λ_effort * ||a||^2 - λ_smooth * ||a - a_prev||^2

Episode termination:
    - terminated when total_score <= success_threshold
    - truncated when t >= max_steps

Notes:
    - 'hold' lets each chosen action persist for multiple physics steps to let pose settle.
    - The DOF order is taken from mujosign.utils.joint_names.DOF_MAP keys (stable).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mujosign.sim_adapter import EnvAdapter
from mujosign.scoring import score as scoring_fn
from mujosign.utils.joint_names import DOF_MAP, TIP_SITES, load_muscle_order


def _as_list_of_specs(specs: Optional[List[Dict[str, Any]]], path_or_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Accept a list of spec dicts, a single file, a directory of JSONs, or None."""
    if specs is not None and len(specs) > 0:
        return [dict(s) for s in specs]
    if path_or_dir:
        p = Path(path_or_dir)
        if p.is_dir():
            return [json.loads(fp.read_text()) for fp in sorted(p.glob("*.json"))]
        elif p.suffix.lower() == ".json" and p.exists():
            return [json.loads(p.read_text())]
    # fallback: empty spec (no targets)
    return [{"gesture": "neutral", "joints": {}}]


def _featurize_spec(spec: Dict[str, Any], dof_order: List[str], normalize_angles: bool = True) -> np.ndarray:
    """
    Build features [target, tolerance, weight] per DOF in a stable order.
    Targets/tolerances are degrees by convention; optionally normalized to ~[-1..1].
    """
    feats: List[float] = []
    joints = spec.get("joints", {})
    scale = (1.0 / 90.0) if normalize_angles else 1.0  # 90° ~ 1.0 helps learning
    for dof in dof_order:
        rule = joints.get(dof, {})
        t = float(rule.get("target", 0.0)) * scale
        tol = float(rule.get("tolerance", 0.0)) * scale
        w = float(rule.get("weight", 1.0))
        feats.extend([t, tol, w])
    return np.asarray(feats, dtype=np.float32)


class PoseActivationEnv(gym.Env):
    """
    Gymnasium environment to learn muscle activations for static hand poses.

    Args:
        env_id: MyoSuite environment id (e.g., "myoHandPoseFixed-v0").
        specs: List of gesture spec dicts (optional if specs_path provided).
        specs_path: File or directory containing gesture JSON(s).
        hold: Number of sim steps to apply each chosen action (pose settle).
        max_steps: Episode length in RL steps (each step holds 'hold' sim steps).
        success_threshold: Terminate early when total score <= this value.
        lambda_effort: Coefficient on ||a||^2 penalty.
        lambda_smooth: Coefficient on ||a - a_prev||^2 penalty.
        normalize_angles: Normalize angles & targets by 90 degrees for learning.
        seed: Random seed.

    Observation space shape:
        feat_dim = 3 * len(DOF_MAP)             # target, tol, weight per DOF
        obs_dim = feat_dim + len(DOF_MAP)       # + current joint angles
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: str = "myoHandPoseFixed-v0",
        specs: Optional[List[Dict[str, Any]]] = None,
        specs_path: Optional[str] = None,
        hold: int = 20,
        max_steps: int = 8,
        success_threshold: float = 1000.0,
        lambda_effort: float = 0.01,
        lambda_smooth: float = 0.0,
        normalize_angles: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        # --- sim & adapter ---
        self._env = gym.make(env_id)
        self._muscle_order = load_muscle_order(env_id)
        self._adapter = EnvAdapter(
            self._env, dof_map=DOF_MAP, tip_sites=TIP_SITES, muscle_order=self._muscle_order
        )

        # --- configuration ---
        self._rng = np.random.RandomState(seed)
        self._dof_order: List[str] = list(DOF_MAP.keys())
        self._specs: List[Dict[str, Any]] = _as_list_of_specs(specs, specs_path)
        self._hold = int(hold)
        self._max_steps = int(max_steps)
        self._success = float(success_threshold)
        self._lam_a = float(lambda_effort)
        self._lam_da = float(lambda_smooth)
        self._normalize = bool(normalize_angles)

        # --- dimensions & spaces ---
        self.nu: int = self._adapter.nu
        self._feat_dim: int = 3 * len(self._dof_order)
        self._obs_dim: int = self._feat_dim + len(self._dof_order)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        # Observations are already scaled near [-2..2], but keep infinite bounds for simplicity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        # --- episode state ---
        self._t: int = 0
        self._spec: Dict[str, Any] = {}
        self._spec_vec = np.zeros((self._feat_dim,), dtype=np.float32)
        self._a_prev = np.zeros((self.nu,), dtype=np.float32)

    # -------------- helpers --------------

    def _current_angles_vec(self) -> np.ndarray:
        angles_deg = self._adapter.get_joint_angles_deg()
        vec = np.asarray([float(angles_deg.get(d, 0.0)) for d in self._dof_order], dtype=np.float32)
        if self._normalize:
            vec = vec * (1.0 / 90.0)
        return vec

    def _obs(self) -> np.ndarray:
        return np.concatenate([self._spec_vec, self._current_angles_vec()], axis=0).astype(np.float32)

    # -------------- Gymnasium API --------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)
        self._t = 0
        # sample spec
        if len(self._specs) == 1:
            self._spec = dict(self._specs[0])
        else:
            self._spec = dict(self._rng.choice(self._specs))
        self._spec_vec = _featurize_spec(self._spec, self._dof_order, normalize_angles=self._normalize)

        # reset sim
        self._adapter.reset()
        self._a_prev.fill(0.0)

        return self._obs(), {}

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(self.nu)
        a = np.clip(a, 0.0, 1.0)

        # hold constant action to let pose settle
        for _ in range(self._hold):
            _, _, terminated, truncated, _ = self._adapter.step(a)
            if terminated or truncated:
                self._adapter.reset()

        # score current state vs spec
        angles_deg = self._adapter.get_joint_angles_deg()
        breakdown = scoring_fn(angles_deg, self._spec)
        total = float(breakdown["total"])

        # reward shaping
        effort = float(np.sum(a * a))
        smooth = float(np.sum((a - self._a_prev) ** 2))
        reward = - total - self._lam_a * effort - self._lam_da * smooth

        # bookkeeping
        self._a_prev = a
        self._t += 1
        terminated = bool(total <= self._success)
        truncated = bool(self._t >= self._max_steps)

        info = {
            "score": breakdown,
            "spec": self._spec,
            "effort": effort,
            "smooth": smooth,
        }
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    # -------------- convenience --------------

    @property
    def dof_order(self) -> List[str]:
        """Stable DOF order used for features and angle vectors."""
        return list(self._dof_order)

    def set_spec(self, spec: Dict[str, Any]):
        """Force a specific spec for subsequent episodes (useful for evaluation)."""
        self._specs = [dict(spec)]
        self._spec = dict(spec)
        self._spec_vec = _featurize_spec(self._spec, self._dof_order, normalize_angles=self._normalize)

    def set_specs_from(self, path_or_dir: str):
        """Load one or many JSON gesture specs from a path or directory."""
        self._specs = _as_list_of_specs(None, path_or_dir)
# src/mujosign/sim_adapter.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

class EnvAdapter:
    """
    Thin wrapper around a Gymnasium MyoSuite env that exposes:
      - joint angles in degrees by spec DOF name (via DOF_MAP)
      - fingertip positions (via TIP_SITES)
      - reset/step passthrough
    """
    def __init__(self, env, dof_map: Dict[str, str], tip_sites: Dict[str, str], muscle_order: Optional[List[str]] = None):
        self.env = env
        self.sim = env.unwrapped.sim
        self.model = self.sim.model
        self.data = self.sim.data
        self.dof_map = dict(dof_map)
        self.tip_sites = dict(tip_sites)
        self.nu = int(getattr(self.model, "nu", 0))
        self.muscle_order = list(muscle_order) if muscle_order else None

        # Build name -> joint index & qpos address
        # Use name_jntadr/name buffer if needed
        names_buf = getattr(self.model, "names", None)
        name_jntadr = getattr(self.model, "name_jntadr", None)
        joint_names = []
        if names_buf is not None and name_jntadr is not None:
            def _decode(start):
                end = start
                if isinstance(names_buf, (bytes, bytearray)):
                    z = names_buf.find(b"\x00", start)
                    end = len(names_buf) if z == -1 else z
                    return names_buf[start:end].decode("utf-8", "ignore")
                # numpy path
                while end < len(names_buf) and names_buf[end] != 0:
                    end += 1
                return bytes(names_buf[start:end]).decode("utf-8", "ignore")
            for j in range(self.model.njnt):
                joint_names.append(_decode(int(name_jntadr[j])))
        else:
            # Fallback: try attribute joint_names if present
            joint_names = [str(n) for n in getattr(self.model, "joint_names", [])]

        self._joint_name_to_qposadr = {}
        for j in range(self.model.njnt):
            name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
            self._joint_name_to_qposadr[name] = int(self.model.jnt_qposadr[j])

        # Site name -> id map (for tips)
        site_names = []
        name_siteadr = getattr(self.model, "name_siteadr", None)
        if names_buf is not None and name_siteadr is not None:
            def _decode(start):
                end = start
                if isinstance(names_buf, (bytes, bytearray)):
                    z = names_buf.find(b"\x00", start)
                    end = len(names_buf) if z == -1 else z
                    return names_buf[start:end].decode("utf-8", "ignore")
                while end < len(names_buf) and names_buf[end] != 0:
                    end += 1
                return bytes(names_buf[start:end]).decode("utf-8", "ignore")
            for i in range(self.model.nsite):
                site_names.append(_decode(int(name_siteadr[i])))
        else:
            site_names = [str(n) for n in getattr(self.model, "site_names", [])]

        self._site_name_to_id = {name: i for i, name in enumerate(site_names)}

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    @staticmethod
    def _deg(x: float) -> float:
        return float(np.degrees(x))

    def get_joint_angles_deg(self) -> Dict[str, float]:
        out = {}
        qpos = self.data.qpos
        for spec_dof, model_joint_name in self.dof_map.items():
            addr = self._joint_name_to_qposadr.get(model_joint_name, None)
            if addr is None:
                continue
            out[spec_dof] = self._deg(qpos[addr])
        return out

    def get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        out = {}
        for key, site_name in self.tip_sites.items():
            sid = self._site_name_to_id.get(site_name, None)
            if sid is None:
                continue
            out[key] = np.array(self.data.site_xpos[sid], dtype=float).copy()  # world meters
        return out
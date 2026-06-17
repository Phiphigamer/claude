"""Particle database with PDG-2023 values for FCC-relevant particles."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Particle:
    name: str
    pdg_id: int
    mass: float        # GeV/c²
    charge: float      # e
    spin: float        # hbar
    lifetime: float    # seconds (np.inf for stable)
    width: float = 0.0 # GeV (decay width)
    color: int = 1     # 1=singlet, 3=triplet, 8=octet
    anti_id: Optional[int] = None
    decay_modes: list = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        return np.isinf(self.lifetime)

    @property
    def ctau(self) -> float:
        """Decay length c*tau in meters."""
        if self.is_stable:
            return np.inf
        return 2.998e8 * self.lifetime

    def __repr__(self) -> str:
        return f"Particle({self.name}, m={self.mass:.4f} GeV)"


class ParticleDB:
    """Particle database with PDG-2023 values."""

    def __init__(self):
        self._particles: dict[int, Particle] = {}
        self._init_db()

    def _init_db(self):
        # Leptons
        self.register(Particle("e-",     11,  0.000511,  -1, 0.5, np.inf))
        self.register(Particle("e+",    -11,  0.000511,  +1, 0.5, np.inf))
        self.register(Particle("mu-",    13,  0.10566,   -1, 0.5, 2.197e-6))
        self.register(Particle("mu+",   -13,  0.10566,   +1, 0.5, 2.197e-6))
        self.register(Particle("tau-",   15,  1.77686,   -1, 0.5, 2.906e-13))
        self.register(Particle("tau+",  -15,  1.77686,   +1, 0.5, 2.906e-13))
        self.register(Particle("nu_e",   12,  0.0,        0, 0.5, np.inf))
        self.register(Particle("nu_mu",  14,  0.0,        0, 0.5, np.inf))
        self.register(Particle("nu_tau", 16,  0.0,        0, 0.5, np.inf))

        # Quarks (constituent masses approximated)
        self.register(Particle("u",   2,  0.00216,  +2/3, 0.5, np.inf, color=3))
        self.register(Particle("u~", -2,  0.00216,  -2/3, 0.5, np.inf, color=3))
        self.register(Particle("d",   1,  0.00467,  -1/3, 0.5, np.inf, color=3))
        self.register(Particle("d~", -1,  0.00467,  +1/3, 0.5, np.inf, color=3))
        self.register(Particle("s",   3,  0.0934,   -1/3, 0.5, np.inf, color=3))
        self.register(Particle("s~", -3,  0.0934,   +1/3, 0.5, np.inf, color=3))
        self.register(Particle("c",   4,  1.27,     +2/3, 0.5, np.inf, color=3))
        self.register(Particle("c~", -4,  1.27,     -2/3, 0.5, np.inf, color=3))
        self.register(Particle("b",   5,  4.18,     -1/3, 0.5, np.inf, color=3))
        self.register(Particle("b~", -5,  4.18,     +1/3, 0.5, np.inf, color=3))
        self.register(Particle("t",   6,  172.69,   +2/3, 0.5, 5e-25,  color=3, width=1.42))
        self.register(Particle("t~", -6,  172.69,   -2/3, 0.5, 5e-25,  color=3, width=1.42))

        # Anti-neutrinos
        self.register(Particle("nu_e~",   -12, 0.0, 0, 0.5, np.inf))
        self.register(Particle("nu_mu~",  -14, 0.0, 0, 0.5, np.inf))
        self.register(Particle("nu_tau~", -16, 0.0, 0, 0.5, np.inf))

        # Gluon
        self.register(Particle("g", 21, 0.0, 0, 1, np.inf, color=8))

        # Gauge bosons
        self.register(Particle("gamma", 22, 0.0,      0, 1, np.inf))
        self.register(Particle("Z",     23, 91.1876,  0, 1, np.inf, width=2.4952,
            decay_modes=[
                (0.2000, [11, -11]),   # e+e-
                (0.2000, [13, -13]),   # mu+mu-
                (0.2000, [15, -15]),   # tau+tau-
                (0.0670, [12, -12]),   # nu_e
                (0.0670, [14, -14]),   # nu_mu
                (0.0670, [16, -16]),   # nu_tau
                (0.1560, [2, -2]),     # uu-bar
                (0.1230, [1, -1]),     # dd-bar
            ]))
        self.register(Particle("W+", 24,  80.377,  +1, 1, np.inf, width=2.085,
            decay_modes=[
                (0.1071, [11, -12]),   # e+ nu_e
                (0.1071, [13, -14]),   # mu+ nu_mu
                (0.1071, [15, -16]),   # tau+ nu_tau
                (0.6787, [2, -1]),     # ud-bar (approx hadronic)
            ]))
        self.register(Particle("W-", -24, 80.377, -1, 1, np.inf, width=2.085))

        # Higgs boson (PDG 2023; cc-bar added to reach ~1.0)
        self.register(Particle("H",  25, 125.25, 0, 0, np.inf, width=4.07e-3,
            decay_modes=[
                (0.5824, [5, -5]),     # bb-bar (dominant)
                (0.2137, [24, -24]),   # WW*
                (0.0827, [21, 21]),    # gg (via loop)
                (0.0634, [4, -4]),     # cc-bar
                (0.0260, [15, -15]),   # tau+tau-
                (0.0272, [23, 23]),    # ZZ*
                (0.0023, [13, -13]),   # mu+mu-
                (0.0023, [22, 22]),    # gamma gamma
            ]))

    def register(self, particle: Particle):
        self._particles[particle.pdg_id] = particle

    def get(self, pdg_id: int) -> Particle:
        if pdg_id not in self._particles:
            raise KeyError(f"Particle with PDG ID {pdg_id} not found")
        return self._particles[pdg_id]

    def get_by_name(self, name: str) -> Particle:
        for p in self._particles.values():
            if p.name == name:
                return p
        raise KeyError(f"Particle '{name}' not found")

    def list_particles(self) -> list[Particle]:
        return list(self._particles.values())

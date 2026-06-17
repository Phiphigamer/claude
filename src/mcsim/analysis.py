"""Analysis tools: histogramming, observable computation."""
import numpy as np
from dataclasses import dataclass
from .kinematics import FourMomentum, invariant_mass
from .generator import Event


@dataclass
class Histogram:
    """Simple 1D histogram."""
    edges: np.ndarray
    counts: np.ndarray
    sumw2: np.ndarray  # sum of weights squared

    @classmethod
    def create(cls, bins: int, low: float, high: float) -> "Histogram":
        edges = np.linspace(low, high, bins + 1)
        return cls(edges, np.zeros(bins), np.zeros(bins))

    def fill(self, values: np.ndarray | float, weights: np.ndarray | float = 1.0):
        c, _ = np.histogram(values, bins=self.edges, weights=weights)
        self.counts += c
        w2, _ = np.histogram(values, bins=self.edges, weights=np.ones_like(np.atleast_1d(values)) * weights)
        self.sumw2 += w2

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.sumw2)

    @property
    def bin_width(self) -> float:
        return self.edges[1] - self.edges[0]


class EventAnalyzer:
    """Compute physics observables from events."""

    def __init__(self, events: list[Event]):
        self.events = events

    def invariant_masses(self, pdg_ids: list[int]) -> np.ndarray:
        """Compute invariant mass of specified particles in each event."""
        masses = []
        for ev in self.events:
            selected = [p for pid, p in ev.final_state if pid in pdg_ids or -pid in pdg_ids]
            if len(selected) >= 2:
                masses.append(invariant_mass(*selected))
            elif len(selected) == 1:
                masses.append(selected[0].mass)
        return np.array(masses)

    def pt_spectrum(self, pdg_id: int) -> np.ndarray:
        """Collect pT values for given particle type."""
        pts = []
        for ev in self.events:
            for pid, p in ev.final_state:
                if abs(pid) == abs(pdg_id):
                    pts.append(p.pt)
        return np.array(pts)

    def eta_distribution(self, pdg_id: int) -> np.ndarray:
        """Collect pseudorapidity values."""
        etas = []
        for ev in self.events:
            for pid, p in ev.final_state:
                if abs(pid) == abs(pdg_id):
                    etas.append(p.eta)
        return np.array(etas)

    def cos_theta_distribution(self, pdg_id: int) -> np.ndarray:
        """cos(θ) of particle direction wrt beam axis."""
        cos_thetas = []
        for ev in self.events:
            for pid, p in ev.final_state:
                if abs(pid) == abs(pdg_id):
                    cos_thetas.append(p.pz / max(p.p, 1e-30))
        return np.array(cos_thetas)

    def forward_backward_asymmetry(self, pdg_id: int) -> float:
        """Compute AFB for given particle type."""
        cos_theta = self.cos_theta_distribution(pdg_id)
        if len(cos_theta) == 0:
            return 0.0
        N_F = np.sum(cos_theta > 0)
        N_B = np.sum(cos_theta < 0)
        total = N_F + N_B
        if total == 0:
            return 0.0
        return (N_F - N_B) / total

    def decay_mode_fractions(self) -> dict[str, float]:
        """Count fraction of events per decay mode."""
        from collections import Counter
        modes = Counter()
        for ev in self.events:
            if "Z_daughters" in ev.metadata:
                key = str(sorted(ev.metadata["Z_daughters"]))
            elif "process" in ev.metadata:
                key = ev.metadata["process"]
            else:
                key = "unknown"
            modes[key] += 1
        total = sum(modes.values())
        return {k: v / total for k, v in modes.items()}

    def missing_energy(self) -> np.ndarray:
        """Compute missing energy (from neutrinos) per event."""
        missing = []
        for ev in self.events:
            E_vis = sum(p.E for _, p in ev.final_state
                        if abs(_) not in {12, 14, 16})
            missing.append(ev.ecm - E_vis)
        return np.array(missing)

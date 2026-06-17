"""Event generator: orchestrates physics processes into full events."""
import numpy as np
from dataclasses import dataclass, field
from .particles import ParticleDB
from .kinematics import FourMomentum, rambo, boost, invariant_mass
from .processes import Decay, BreitWigner


@dataclass
class Event:
    """A single simulated event."""
    event_id: int
    ecm: float
    initial_state: list[tuple[int, FourMomentum]] = field(default_factory=list)
    final_state: list[tuple[int, FourMomentum]] = field(default_factory=list)
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def get_momenta(self, pdg_id: int) -> list[FourMomentum]:
        return [p for pid, p in self.final_state if pid == abs(pdg_id)]

    def invariant_mass_all(self) -> float:
        return invariant_mass(*[p for _, p in self.final_state])


class EventGenerator:
    """
    Main event generator for FCC-ee processes.
    Supports: Z → ff, H decays, e+e- → ZH.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.db = ParticleDB()
        self._event_count = 0

    def generate_z_decay(self, ecm: float = 91.1876) -> Event:
        """
        Generate e+e- → Z → ff event at given ECM.
        Uses Breit-Wigner sampling for Z mass.
        """
        Z = self.db.get(23)
        bw = BreitWigner(Z.mass, Z.width)
        mZ_sampled = bw.sample_mass(
            max(2.0, ecm - 5 * Z.width),
            ecm + 5 * Z.width,
            self.rng
        )
        mZ_sampled = min(mZ_sampled, ecm)

        p_Z = FourMomentum.from_mass_at_rest(mZ_sampled)
        decay = Decay(Z, self.db, self.rng)
        pdg_ids, momenta = decay.generate(p_Z)

        event = Event(
            event_id=self._event_count,
            ecm=ecm,
            initial_state=[(11, FourMomentum(ecm/2, 0, 0, ecm/2)),
                           (-11, FourMomentum(ecm/2, 0, 0, -ecm/2))],
            final_state=list(zip(pdg_ids, momenta)),
            metadata={"process": "e+e- -> Z -> ff", "mZ_sampled": mZ_sampled}
        )
        self._event_count += 1
        return event

    def generate_higgs_decay(self, ecm: float = 125.25) -> Event:
        """Generate H → XX decay event."""
        H = self.db.get(25)
        p_H = FourMomentum.from_mass_at_rest(H.mass)
        decay = Decay(H, self.db, self.rng)
        pdg_ids, momenta = decay.generate(p_H)

        event = Event(
            event_id=self._event_count,
            ecm=ecm,
            initial_state=[],
            final_state=list(zip(pdg_ids, momenta)),
            metadata={"process": "H -> XX"}
        )
        self._event_count += 1
        return event

    def generate_zh_event(self, ecm: float = 240.0) -> Event:
        """
        Generate e+e- → ZH event, then decay Z and H independently.
        """
        mH = self.db.get(25).mass
        mZ = self.db.get(23).mass

        if ecm < mZ + mH:
            raise ValueError(f"ECM {ecm} GeV too low for ZH production (need >{mZ+mH:.1f} GeV)")

        # Generate ZH in CM frame using 2-body phase space
        zh_momenta = rambo(2, ecm, [mZ, mH], self.rng)
        p_Z, p_H = zh_momenta

        # Decay Z
        Z = self.db.get(23)
        decay_Z = Decay(Z, self.db, self.rng)
        z_ids, z_momenta = decay_Z.generate(p_Z)

        # Decay H
        H = self.db.get(25)
        decay_H = Decay(H, self.db, self.rng)
        h_ids, h_momenta = decay_H.generate(p_H)

        final_state = list(zip(z_ids, z_momenta)) + list(zip(h_ids, h_momenta))

        event = Event(
            event_id=self._event_count,
            ecm=ecm,
            initial_state=[(11, FourMomentum(ecm/2, 0, 0, ecm/2)),
                           (-11, FourMomentum(ecm/2, 0, 0, -ecm/2))],
            final_state=final_state,
            metadata={
                "process": "e+e- -> ZH",
                "Z_daughters": z_ids,
                "H_daughters": h_ids,
            }
        )
        self._event_count += 1
        return event

    def run(self, process: str, n_events: int, ecm: float = None, **kwargs) -> list[Event]:
        """
        Generate n_events for given process.
        process: 'z_decay', 'higgs_decay', 'zh'
        """
        generators = {
            "z_decay":     (self.generate_z_decay,   91.1876),
            "higgs_decay": (self.generate_higgs_decay, 125.25),
            "zh":          (self.generate_zh_event,   240.0),
        }
        if process not in generators:
            raise ValueError(f"Unknown process '{process}'. Choose from: {list(generators)}")

        gen_fn, default_ecm = generators[process]
        if ecm is None:
            ecm = default_ecm

        return [gen_fn(ecm=ecm, **kwargs) for _ in range(n_events)]

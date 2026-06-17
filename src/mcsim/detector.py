"""Simple parametric detector simulation with Gaussian smearing."""
import numpy as np
from .kinematics import FourMomentum


class Detector:
    """
    Simplified FCC-ee detector model (IDEA-like).
    Applies Gaussian resolution effects to final-state particles.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        # Resolution parameters (sigma/E or sigma_E depending on type)
        self.em_res_a = 0.03    # EM calorimeter: σ/E = a/√E
        self.had_res_a = 0.50   # Hadronic: σ/E = a/√E
        self.track_res = 1e-4   # Track momentum: σ(p)/p² in GeV^-1
        self.angular_res = 1e-4 # Angular resolution in radians

    def _is_charged_lepton(self, pdg_id: int) -> bool:
        return abs(pdg_id) in {11, 13, 15}

    def _is_neutrino(self, pdg_id: int) -> bool:
        return abs(pdg_id) in {12, 14, 16}

    def _is_photon(self, pdg_id: int) -> bool:
        return abs(pdg_id) == 22

    def _is_quark_gluon(self, pdg_id: int) -> bool:
        return abs(pdg_id) in {1, 2, 3, 4, 5, 6, 21}

    def smear_lepton(self, p: FourMomentum) -> FourMomentum:
        """Apply track momentum smearing to charged lepton."""
        p_mag = p.p
        sigma_p = self.track_res * p_mag**2
        delta_p = self.rng.normal(0, sigma_p)
        scale = (p_mag + delta_p) / max(p_mag, 1e-10)

        # Angular smearing
        theta = p.theta
        phi = p.phi
        theta_s = theta + self.rng.normal(0, self.angular_res)
        phi_s = phi + self.rng.normal(0, self.angular_res)
        p_new = (p_mag + delta_p)

        return FourMomentum.from_mass_and_momentum(
            p.mass,
            p_new * np.sin(theta_s) * np.cos(phi_s),
            p_new * np.sin(theta_s) * np.sin(phi_s),
            p_new * np.cos(theta_s),
        )

    def smear_photon(self, p: FourMomentum) -> FourMomentum:
        """Apply EM calorimeter smearing to photon."""
        E = p.E
        sigma_E = self.em_res_a * np.sqrt(E)
        E_new = max(E + self.rng.normal(0, sigma_E), 0.0)
        scale = E_new / max(E, 1e-10)
        return FourMomentum(E_new, scale * p.px, scale * p.py, scale * p.pz)

    def smear_jet(self, p: FourMomentum) -> FourMomentum:
        """Apply hadronic calorimeter smearing (jet proxy)."""
        E = p.E
        sigma_E = self.had_res_a * np.sqrt(E)
        E_new = max(E + self.rng.normal(0, sigma_E), 0.0)
        scale = E_new / max(E, 1e-10)
        return FourMomentum(E_new, scale * p.px, scale * p.py, scale * p.pz)

    def reconstruct(self, pdg_id: int, p: FourMomentum) -> FourMomentum | None:
        """
        Apply detector effects. Returns None for undetected particles (neutrinos).
        """
        if self._is_neutrino(pdg_id):
            return None
        if self._is_charged_lepton(pdg_id):
            return self.smear_lepton(p)
        if self._is_photon(pdg_id):
            return self.smear_photon(p)
        # Quarks/gluons → jets
        return self.smear_jet(p)

    def process_event(self, final_state: list[tuple[int, FourMomentum]]) -> list[tuple[int, FourMomentum]]:
        """Apply detector simulation to full event final state."""
        result = []
        for pdg_id, p in final_state:
            p_reco = self.reconstruct(pdg_id, p)
            if p_reco is not None:
                result.append((pdg_id, p_reco))
        return result

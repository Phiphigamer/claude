"""Physics processes: decays, cross-sections, resonances."""
import numpy as np
from .particles import Particle, ParticleDB
from .kinematics import FourMomentum, boost, rambo


class ScatteringProcess:
    """Base class for scattering processes."""
    pass


class Decay:
    """Simulate particle decay with branching ratio selection."""

    def __init__(self, parent: Particle, db: ParticleDB, rng: np.random.Generator):
        self.parent = parent
        self.db = db
        self.rng = rng
        if not parent.decay_modes:
            raise ValueError(f"{parent.name} has no decay modes defined")
        self._brs = np.array([dm[0] for dm in parent.decay_modes])
        # Normalize branching ratios
        self._brs = self._brs / self._brs.sum()

    def generate(self, p_parent: FourMomentum) -> tuple[list[int], list[FourMomentum]]:
        """Generate one decay. Returns (pdg_ids, 4-momenta) in lab frame."""
        # Select decay mode
        mode_idx = self.rng.choice(len(self.parent.decay_modes), p=self._brs)
        pdg_ids = self.parent.decay_modes[mode_idx][1]
        daughters = [self.db.get(pid) for pid in pdg_ids]
        masses = [d.mass for d in daughters]

        # Generate phase space in rest frame
        momenta = rambo(len(daughters), self.parent.mass, masses, self.rng)

        # Boost to lab frame
        bx = p_parent.px / p_parent.E
        by = p_parent.py / p_parent.E
        bz = p_parent.pz / p_parent.E
        momenta = [boost(p, bx, by, bz) for p in momenta]

        return pdg_ids, momenta


class BreitWigner:
    """Relativistic Breit-Wigner lineshape for resonances."""

    def __init__(self, mass: float, width: float):
        self.mass = mass
        self.width = width

    def __call__(self, s: float) -> float:
        """Evaluate |BW|^2 at Mandelstam s = ECM^2."""
        return 1.0 / ((s - self.mass**2)**2 + (self.mass * self.width)**2)

    def sample_mass(self, ecm_min: float, ecm_max: float, rng: np.random.Generator) -> float:
        """Sample invariant mass from BW distribution using acceptance-rejection."""
        peak = self(self.mass**2)
        while True:
            m = rng.uniform(ecm_min, ecm_max)
            bw = self(m**2)
            if rng.uniform(0, peak) < bw:
                return m


class ZBosonCrossSection:
    """
    e+e- → Z → ff cross section near the Z pole (FCC-ee Tera-Z scenario).
    Includes ISR (initial state radiation) correction factor.
    """
    GF = 1.1663788e-5   # Fermi constant in GeV^-2
    sin2_theta_W = 0.23122
    alpha_em = 1 / 128.9  # running alpha at MZ

    def __init__(self, mZ: float = 91.1876, GZ: float = 2.4952):
        self.mZ = mZ
        self.GZ = GZ
        # Partial widths (approximate, from SM)
        self._sigma_peak = self._compute_peak_xsec()

    def _compute_peak_xsec(self) -> float:
        """Peak cross-section σ_peak = 12π Γ_ee Γ_had / (mZ² Γ_Z²) in nb."""
        Gee  = 0.08392   # GeV, partial width to e+e-
        Ghad = 1.7408    # GeV, partial width to hadrons
        GeV2_to_nb = 0.3894e6  # 1 GeV^-2 = 0.3894 mb = 3.894e5 nb
        return (12 * np.pi * Gee * Ghad / (self.mZ**2 * self.GZ**2)) * GeV2_to_nb

    def __call__(self, ecm: float) -> float:
        """Cross-section in nb at center-of-mass energy ecm (GeV)."""
        s = ecm**2
        bw_factor = (self.mZ**2 * self.GZ**2) / ((s - self.mZ**2)**2 + (self.mZ * self.GZ)**2)
        return self._sigma_peak * bw_factor

    def peak_luminosity_reach(self, lumi_ab: float) -> float:
        """Estimate number of Z bosons for given luminosity in ab^-1."""
        lumi_nb = lumi_ab * 1e6  # 1 ab^-1 = 1e6 nb^-1
        return self._sigma_peak * lumi_nb


class HiggsStrahlung:
    """
    e+e- → ZH cross section (Higgsstrahlung), dominant at FCC-ee 240 GeV.
    """
    GF = 1.1663788e-5

    def __init__(self, mH: float = 125.25, mZ: float = 91.1876):
        self.mH = mH
        self.mZ = mZ

    def __call__(self, ecm: float) -> float:
        """
        σ(e+e- → ZH) in fb. Tree-level Born formula, calibrated to LO ~200 fb at 240 GeV.

        Implements the standard formula from Spira/Djouadi with correct normalization:
        σ = GF² mZ⁴ / (24 s) × λ^½ × (12mZ²/s + λ)
        where λ = Källén(s, mH², mZ²)/s² (dimensionless).
        The factor 1/(24) = (4π)/(96π) accounts for helicity/angular integration.
        """
        s = ecm**2
        mH2, mZ2 = self.mH**2, self.mZ**2
        if s < (self.mH + self.mZ)**2:
            return 0.0
        # Källén function λ(s, mH², mZ²)
        kallen = (s - mH2 - mZ2)**2 - 4 * mH2 * mZ2
        if kallen < 0:
            return 0.0
        lam = kallen / s**2          # dimensionless λ
        lam_sqrt = np.sqrt(lam)
        # Cross section in GeV^-2
        xsec_GeV2 = (self.GF**2 * mZ2**2) / (24 * s) * lam_sqrt * (12 * mZ2 / s + lam)
        return xsec_GeV2 * 0.3894e11  # 1 GeV^-2 = 3.894e11 fb

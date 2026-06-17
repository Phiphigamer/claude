"""Relativistic kinematics: 4-vectors, boosts, phase space."""
import numpy as np
from dataclasses import dataclass


@dataclass
class FourMomentum:
    """4-momentum vector (E, px, py, pz) in GeV."""
    E: float
    px: float
    py: float
    pz: float

    @classmethod
    def from_mass_and_momentum(cls, mass: float, px: float, py: float, pz: float) -> "FourMomentum":
        E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
        return cls(E, px, py, pz)

    @classmethod
    def from_mass_at_rest(cls, mass: float) -> "FourMomentum":
        return cls(mass, 0.0, 0.0, 0.0)

    @classmethod
    def from_massless(cls, energy: float, theta: float, phi: float) -> "FourMomentum":
        px = energy * np.sin(theta) * np.cos(phi)
        py = energy * np.sin(theta) * np.sin(phi)
        pz = energy * np.cos(theta)
        return cls(energy, px, py, pz)

    @property
    def mass(self) -> float:
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return np.sqrt(max(m2, 0.0))

    @property
    def p(self) -> float:
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def pt(self) -> float:
        return np.sqrt(self.px**2 + self.py**2)

    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        p_mag = self.p
        if p_mag == 0:
            return 0.0
        cos_theta = self.pz / p_mag
        if abs(cos_theta) >= 1.0:
            return np.sign(cos_theta) * 1e9
        return -0.5 * np.log((1 - cos_theta) / (1 + cos_theta))

    @property
    def phi(self) -> float:
        return np.arctan2(self.py, self.px)

    @property
    def theta(self) -> float:
        return np.arccos(np.clip(self.pz / max(self.p, 1e-30), -1, 1))

    @property
    def rapidity(self) -> float:
        if self.E + self.pz <= 0 or self.E - self.pz <= 0:
            return 0.0
        return 0.5 * np.log((self.E + self.pz) / (self.E - self.pz))

    @property
    def beta(self) -> float:
        return self.p / self.E

    @property
    def gamma(self) -> float:
        return self.E / max(self.mass, 1e-30)

    def boost_to_rest(self) -> "FourMomentum":
        """Boost this vector to the rest frame of itself."""
        return boost(self, -self.px / self.E, -self.py / self.E, -self.pz / self.E)

    def __add__(self, other: "FourMomentum") -> "FourMomentum":
        return FourMomentum(self.E + other.E, self.px + other.px,
                            self.py + other.py, self.pz + other.pz)

    def __sub__(self, other: "FourMomentum") -> "FourMomentum":
        return FourMomentum(self.E - other.E, self.px - other.px,
                            self.py - other.py, self.pz - other.pz)

    def __repr__(self) -> str:
        return f"FourMomentum(E={self.E:.4f}, px={self.px:.4f}, py={self.py:.4f}, pz={self.pz:.4f}, m={self.mass:.4f})"


def boost(p: FourMomentum, bx: float, by: float, bz: float) -> FourMomentum:
    """Apply a Lorentz boost (bx, by, bz) = beta vector to a 4-momentum."""
    b2 = bx**2 + by**2 + bz**2
    if b2 == 0:
        return p
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = bx * p.px + by * p.py + bz * p.pz
    gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0

    new_E  = gamma * (p.E - bp)
    new_px = p.px + gamma2 * bp * bx - gamma * bx * p.E
    new_py = p.py + gamma2 * bp * by - gamma * by * p.E
    new_pz = p.pz + gamma2 * bp * bz - gamma * bz * p.E
    return FourMomentum(new_E, new_px, new_py, new_pz)


def invariant_mass(*momenta: FourMomentum) -> float:
    """Compute invariant mass of a system of 4-momenta."""
    total = FourMomentum(0, 0, 0, 0)
    for p in momenta:
        total = total + p
    return total.mass


def dot(p1: FourMomentum, p2: FourMomentum) -> float:
    """Minkowski dot product (metric +---)."""
    return p1.E * p2.E - p1.px * p2.px - p1.py * p2.py - p1.pz * p2.pz


def rambo(n: int, ecm: float, masses: list[float], rng: np.random.Generator) -> list[FourMomentum]:
    """
    RAMBO algorithm: uniform n-body phase space generation.
    Generates n massless momenta and maps to massive case.
    Returns list of FourMomentum for n final-state particles.
    """
    # Step 1: Generate massless momenta isotropically
    q_list = []
    for _ in range(n):
        r1, r2, r3, r4 = rng.uniform(size=4)
        cos_theta = 2 * r1 - 1
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2 * np.pi * r2
        E = -np.log(r3 * r4)
        px = E * sin_theta * np.cos(phi)
        py = E * sin_theta * np.sin(phi)
        pz = E * cos_theta
        q_list.append(FourMomentum(E, px, py, pz))

    # Step 2: Boost all momenta to the rest frame of Q = sum of q_i
    Q = FourMomentum(0, 0, 0, 0)
    for q in q_list:
        Q = Q + q

    M = Q.mass
    # Boost into frame where Q is at rest: beta = Q.p/Q.E
    bx, by, bz = Q.px / Q.E, Q.py / Q.E, Q.pz / Q.E
    p_list = [boost(q, bx, by, bz) for q in q_list]

    # Step 3: Scale to ECM
    x = ecm / M
    p_list = [FourMomentum(x * p.E, x * p.px, x * p.py, x * p.pz) for p in p_list]

    # Step 4: Correct for masses (iterative rescaling)
    if all(m == 0 for m in masses):
        return p_list

    p_list = _correct_masses(p_list, masses, ecm)
    return p_list


def _correct_masses(p_list: list[FourMomentum], masses: list[float], ecm: float) -> list[FourMomentum]:
    """Correct massless momenta for final-state masses using iterative xi-finding."""
    n = len(p_list)
    p_mags = [p.p for p in p_list]

    # Find xi such that sum_i sqrt(masses[i]^2 + xi^2 * |p_i|^2) = ecm
    def f(xi):
        return sum(np.sqrt(masses[i]**2 + xi**2 * p_mags[i]**2) for i in range(n)) - ecm

    # Bisection
    xi_lo, xi_hi = 0.0, ecm / max(sum(p_mags), 1e-30)
    if f(xi_hi) < 0:
        # Can't fit masses into ECM, return massless
        return p_list

    for _ in range(50):
        xi_mid = (xi_lo + xi_hi) / 2
        if f(xi_mid) < 0:
            xi_lo = xi_mid
        else:
            xi_hi = xi_mid

    xi = (xi_lo + xi_hi) / 2
    result = []
    for i, p in enumerate(p_list):
        new_p_mag = xi * p_mags[i]
        new_E = np.sqrt(masses[i]**2 + new_p_mag**2)
        scale = new_p_mag / max(p_mags[i], 1e-30)
        result.append(FourMomentum(new_E, scale * p.px, scale * p.py, scale * p.pz))
    return result

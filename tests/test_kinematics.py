"""Tests for 4-vector kinematics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from mcsim.kinematics import FourMomentum, boost, invariant_mass, rambo


def test_mass_at_rest():
    p = FourMomentum.from_mass_at_rest(91.1876)
    assert abs(p.mass - 91.1876) < 1e-6


def test_massless_particle():
    p = FourMomentum(10, 0, 0, 10)
    assert abs(p.mass) < 1e-6


def test_invariant_mass_two_body():
    p1 = FourMomentum(50, 0, 0, 50)
    p2 = FourMomentum(50, 0, 0, -50)
    m = invariant_mass(p1, p2)
    assert abs(m - 100) < 1e-6


def test_boost_roundtrip():
    p = FourMomentum.from_mass_and_momentum(91.1876, 10, 5, 20)
    bx, by, bz = 0.1, 0.05, 0.2
    p_boosted = boost(p, bx, by, bz)
    p_back = boost(p_boosted, -bx, -by, -bz)
    assert abs(p_back.E - p.E) < 1e-7
    assert abs(p_back.px - p.px) < 1e-7


def test_rambo_energy_conservation():
    rng = np.random.default_rng(42)
    ecm = 91.1876
    momenta = rambo(3, ecm, [0.0, 0.0, 0.0], rng)
    total_E = sum(p.E for p in momenta)
    assert abs(total_E - ecm) < 1e-5


def test_rambo_momentum_conservation():
    rng = np.random.default_rng(42)
    ecm = 91.1876
    momenta = rambo(4, ecm, [0.0] * 4, rng)
    assert abs(sum(p.px for p in momenta)) < 1e-5
    assert abs(sum(p.py for p in momenta)) < 1e-5
    assert abs(sum(p.pz for p in momenta)) < 1e-5


def test_four_momentum_addition():
    p1 = FourMomentum(10, 1, 2, 3)
    p2 = FourMomentum(20, -1, 3, -3)
    total = p1 + p2
    assert abs(total.E - 30) < 1e-10
    assert abs(total.px - 0) < 1e-10

"""Tests for physics processes and event generator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from mcsim.particles import ParticleDB
from mcsim.kinematics import FourMomentum, invariant_mass
from mcsim.processes import Decay, ZBosonCrossSection, HiggsStrahlung
from mcsim.generator import EventGenerator


def test_z_decay_energy_conservation():
    db = ParticleDB()
    Z = db.get(23)
    rng = np.random.default_rng(0)
    p_Z = FourMomentum.from_mass_at_rest(Z.mass)
    decay = Decay(Z, db, rng)
    _, momenta = decay.generate(p_Z)
    total_E = sum(p.E for p in momenta)
    assert abs(total_E - Z.mass) < 1e-4


def test_z_xsec_peak():
    xsec = ZBosonCrossSection()
    sigma_peak = xsec(91.1876)
    assert sigma_peak > 10  # ~40 nb at peak


def test_z_xsec_far_from_peak():
    xsec = ZBosonCrossSection()
    assert xsec(100.0) < xsec(91.1876)


def test_zh_xsec_threshold():
    zh = HiggsStrahlung()
    assert zh(200.0) == 0.0      # below threshold
    assert zh(240.0) > 0.0       # above threshold
    assert zh(240.0) > zh(300.0) # peaks near threshold


def test_event_generator_z_decay():
    gen = EventGenerator(seed=1)
    events = gen.run("z_decay", 50, ecm=91.1876)
    assert len(events) == 50
    for ev in events:
        assert len(ev.final_state) >= 2
        total_E = sum(p.E for _, p in ev.final_state)
        # Visible energy should be < ECM (some goes to BW off-shell)
        assert total_E > 0


def test_event_generator_zh():
    gen = EventGenerator(seed=2)
    events = gen.run("zh", 10, ecm=240.0)
    assert len(events) == 10
    for ev in events:
        assert len(ev.final_state) >= 4
        assert "Z_daughters" in ev.metadata
        assert "H_daughters" in ev.metadata


def test_event_generator_higgs_decay():
    gen = EventGenerator(seed=3)
    events = gen.run("higgs_decay", 20)
    assert len(events) == 20
    for ev in events:
        assert len(ev.final_state) >= 2

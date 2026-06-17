"""Tests for particle database."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from mcsim.particles import Particle, ParticleDB


def test_electron():
    db = ParticleDB()
    e = db.get(11)
    assert e.name == "e-"
    assert abs(e.mass - 0.000511) < 1e-6
    assert e.is_stable


def test_z_boson():
    db = ParticleDB()
    Z = db.get(23)
    assert abs(Z.mass - 91.1876) < 0.001
    assert abs(Z.width - 2.4952) < 0.001
    assert len(Z.decay_modes) > 0


def test_higgs_br_sum():
    db = ParticleDB()
    H = db.get(25)
    assert abs(H.mass - 125.25) < 0.01
    br_sum = sum(dm[0] for dm in H.decay_modes)
    assert abs(br_sum - 1.0) < 0.01


def test_get_by_name():
    db = ParticleDB()
    mu = db.get_by_name("mu-")
    assert mu.pdg_id == 13


def test_unknown_particle():
    db = ParticleDB()
    with pytest.raises(KeyError):
        db.get(9999)

#!/usr/bin/env python3
"""FCC-ee Tera-Z: e+e- → Z → ff Monte Carlo simulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcsim import EventGenerator, EventAnalyzer
from mcsim.visualization import (
    plot_angular_distribution, plot_z_lineshape, plot_event_summary,
)
from mcsim.detector import Detector


def main():
    print("=" * 60)
    print("  FCC-ee Tera-Z: e+e- → Z → ff Monte Carlo Simulation")
    print("=" * 60)

    N_EVENTS = 10_000
    ECM = 91.1876

    print(f"\nGenerating {N_EVENTS} events at ECM = {ECM:.4f} GeV...")
    gen = EventGenerator(seed=2024)
    events = gen.run("z_decay", N_EVENTS, ecm=ECM)
    print(f"Generated {len(events)} events.")

    det = Detector(seed=2024)
    for ev in events:
        ev.final_state = det.process_event(ev.final_state)

    analyzer = EventAnalyzer(events)

    print("\n--- Physics Summary ---")
    for lepton_id, name in [(11, "e"), (13, "mu"), (15, "tau")]:
        afb = analyzer.forward_backward_asymmetry(lepton_id)
        print(f"  AFB({name}): {afb:.4f}")

    os.makedirs("plots", exist_ok=True)

    fig1 = plot_z_lineshape()
    fig1.savefig("plots/z_lineshape.png", dpi=150, bbox_inches="tight")
    print("\nSaved: plots/z_lineshape.png")

    cos_theta_e = analyzer.cos_theta_distribution(11)
    if len(cos_theta_e):
        fig2 = plot_angular_distribution(
            cos_theta_e, title=r"$e^+e^- \to Z \to e^+e^-$: Angular Distribution")
        fig2.savefig("plots/z_angular.png", dpi=150, bbox_inches="tight")
        print("Saved: plots/z_angular.png")

    fig3 = plot_event_summary(events,
        title=f"Z Decay Summary ({N_EVENTS} events, FCC-ee)")
    fig3.savefig("plots/z_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/z_summary.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FCC-ee Higgs run: e+e- → ZH at 240 GeV."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcsim import EventGenerator, EventAnalyzer
from mcsim.visualization import plot_zh_xsec, plot_event_summary
from mcsim.detector import Detector


def main():
    print("=" * 60)
    print("  FCC-ee Higgs Run: e+e- → ZH at 240 GeV")
    print("=" * 60)

    N_EVENTS = 5_000
    ECM = 240.0

    print(f"\nGenerating {N_EVENTS} ZH events at ECM = {ECM} GeV...")
    gen = EventGenerator(seed=2025)
    events = gen.run("zh", N_EVENTS, ecm=ECM)
    print(f"Generated {len(events)} events.")

    det = Detector(seed=2025)
    for ev in events:
        ev.final_state = det.process_event(ev.final_state)

    analyzer = EventAnalyzer(events)
    missing_e = analyzer.missing_energy()
    print(f"\n  Mean missing energy: {missing_e.mean():.2f} GeV")
    print(f"  Std  missing energy: {missing_e.std():.2f} GeV")

    os.makedirs("plots", exist_ok=True)

    fig1 = plot_zh_xsec()
    fig1.savefig("plots/zh_xsec.png", dpi=150, bbox_inches="tight")
    print("\nSaved: plots/zh_xsec.png")

    fig2 = plot_event_summary(events,
        title=f"ZH Production Summary ({N_EVENTS} events, √s={ECM} GeV)")
    fig2.savefig("plots/zh_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/zh_summary.png")

    fig3, ax = plt.subplots(figsize=(7, 5))
    ax.hist(missing_e, bins=50, color="#9467bd", alpha=0.75)
    ax.set_xlabel("Missing Energy [GeV]", fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(r"$e^+e^- \to ZH$: Missing Energy Distribution", fontsize=13)
    ax.grid(alpha=0.3)
    fig3.savefig("plots/zh_missing_energy.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/zh_missing_energy.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()

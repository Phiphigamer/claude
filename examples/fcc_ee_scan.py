#!/usr/bin/env python3
"""FCC-ee energy scan: Z lineshape & ZH Higgsstrahlung threshold."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcsim.processes import ZBosonCrossSection, HiggsStrahlung


def main():
    print("FCC-ee Energy Scan: Z Lineshape & ZH Threshold")

    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ecm_z = np.linspace(88, 95, 500)
    xsec_z = ZBosonCrossSection()
    sigma_z = np.array([xsec_z(e) for e in ecm_z])

    axes[0].plot(ecm_z, sigma_z, "b-", lw=2)
    axes[0].axvline(91.1876, color="r", ls="--", label=r"$m_Z = 91.19$ GeV")
    axes[0].fill_between(ecm_z, sigma_z, alpha=0.15, color="blue")
    axes[0].set_xlabel(r"$\sqrt{s}$ [GeV]", fontsize=12)
    axes[0].set_ylabel(r"$\sigma(e^+e^- \to Z \to \mathrm{had})$ [nb]", fontsize=11)
    axes[0].set_title("Z Lineshape Scan (FCC-ee Tera-Z)", fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    N_Z = xsec_z.peak_luminosity_reach(150)  # 150 ab^-1 at Tera-Z
    print(f"  Tera-Z yield at peak (150 ab⁻¹): {N_Z:.2e} Z bosons")

    ecm_zh = np.linspace(210, 280, 500)
    zh = HiggsStrahlung()
    sigma_zh = np.array([zh(e) for e in ecm_zh])

    axes[1].plot(ecm_zh, sigma_zh, "g-", lw=2)
    axes[1].axvline(240, color="r", ls="--", label="FCC-ee operating point")
    axes[1].fill_between(ecm_zh, sigma_zh, alpha=0.15, color="green")
    axes[1].set_xlabel(r"$\sqrt{s}$ [GeV]", fontsize=12)
    axes[1].set_ylabel(r"$\sigma(e^+e^- \to ZH)$ [fb]", fontsize=11)
    axes[1].set_title(r"Higgsstrahlung Threshold (FCC-ee Higgs)", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    sigma_at_240 = zh(240)
    print(f"  ZH cross-section at 240 GeV: {sigma_at_240:.1f} fb")

    plt.tight_layout()
    fig.savefig("plots/fcc_ee_scan.png", dpi=150, bbox_inches="tight")
    print("Saved: plots/fcc_ee_scan.png")
    plt.close()


if __name__ == "__main__":
    main()

"""Plotting tools for Monte Carlo simulation results."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .analysis import EventAnalyzer
from .processes import ZBosonCrossSection, HiggsStrahlung


def plot_invariant_mass(
    masses: np.ndarray,
    title: str = "Invariant Mass Distribution",
    bins: int = 60,
    xlabel: str = "m [GeV]",
    color: str = "#1f77b4",
    ax: plt.Axes = None,
    fit_bw: bool = False,
) -> plt.Figure:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    counts, edges = np.histogram(masses, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width * 0.9, color=color, alpha=0.75, label="MC")
    ax.errorbar(centers, counts, yerr=np.sqrt(np.maximum(counts, 1)),
                fmt="none", color="black", linewidth=1)

    if fit_bw and len(masses) > 10:
        from .processes import BreitWigner
        from scipy.optimize import curve_fit
        try:
            bw_obj = BreitWigner(np.mean(masses), 2.5)

            def bw_pdf(m, m0, gamma, norm):
                bw = BreitWigner(m0, gamma)
                return norm * np.array([bw(mi**2) for mi in m])

            mask = counts > 0
            popt, _ = curve_fit(bw_pdf, centers[mask], counts[mask],
                                p0=[np.mean(masses), 2.5, counts.max() * 50])
            m_fit = np.linspace(edges[0], edges[-1], 300)
            ax.plot(m_fit, bw_pdf(m_fit, *popt), "r-", linewidth=2,
                    label=f"BW fit: m={popt[0]:.3f}, Γ={popt[1]:.3f} GeV")
        except Exception:
            pass

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig or ax.figure


def plot_angular_distribution(
    cos_theta: np.ndarray,
    title: str = "Angular Distribution",
    bins: int = 40,
    ax: plt.Axes = None,
) -> plt.Figure:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    counts, edges = np.histogram(cos_theta, bins=bins, range=(-1, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width * 0.9, color="#2ca02c", alpha=0.7, label="MC")
    ax.errorbar(centers, counts, yerr=np.sqrt(np.maximum(counts, 1)),
                fmt="none", color="black", linewidth=1)

    from scipy.optimize import curve_fit
    try:
        def angular_pdf(ct, norm, A):
            return norm * (1 + ct**2 + A * ct)
        popt, _ = curve_fit(angular_pdf, centers, counts, p0=[counts.mean(), 0.1])
        ct_fit = np.linspace(-1, 1, 200)
        ax.plot(ct_fit, angular_pdf(ct_fit, *popt), "r-", lw=2,
                label=f"Fit: AFB={popt[1]/2:.4f}")
    except Exception:
        pass

    ax.set_xlabel(r"$\cos\theta$", fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig or ax.figure


def plot_z_lineshape(ecm_range: tuple = (88, 95), points: int = 200) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ecm_vals = np.linspace(*ecm_range, points)
    xsec = ZBosonCrossSection()
    sigma_vals = [xsec(e) for e in ecm_vals]

    ax.plot(ecm_vals, sigma_vals, "b-", linewidth=2,
            label=r"$\sigma(e^+e^- \to Z \to \mathrm{had})$")
    ax.axvline(91.1876, color="r", linestyle="--", alpha=0.7,
               label=r"$m_Z = 91.19$ GeV")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]", fontsize=12)
    ax.set_ylabel(r"$\sigma$ [nb]", fontsize=12)
    ax.set_title("Z Boson Lineshape (FCC-ee Tera-Z)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_zh_xsec(ecm_range: tuple = (200, 365), points: int = 200) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ecm_vals = np.linspace(*ecm_range, points)
    zh = HiggsStrahlung()
    sigma_vals = [zh(e) for e in ecm_vals]

    ax.plot(ecm_vals, sigma_vals, "g-", linewidth=2,
            label=r"$e^+e^- \to ZH$")
    ax.axvline(240, color="r", linestyle="--", alpha=0.7,
               label="FCC-ee operating point (240 GeV)")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]", fontsize=12)
    ax.set_ylabel(r"$\sigma$ [fb]", fontsize=12)
    ax.set_title(r"Higgsstrahlung Cross-Section", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_event_summary(events: list, title: str = "Event Summary") -> plt.Figure:
    from .kinematics import invariant_mass

    analyzer = EventAnalyzer(events)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Invariant mass of visible particles
    ax1 = fig.add_subplot(gs[0, 0])
    all_masses = []
    for ev in events:
        vis = [p for pid, p in ev.final_state if abs(pid) not in {12, 14, 16}]
        if len(vis) >= 2:
            all_masses.append(invariant_mass(*vis))
    if all_masses:
        plot_invariant_mass(np.array(all_masses), title="Visible Inv. Mass",
                            bins=40, ax=ax1)

    # 2. Missing energy
    ax2 = fig.add_subplot(gs[0, 1])
    missing = analyzer.missing_energy()
    ax2.hist(missing, bins=40, color="#d62728", alpha=0.7)
    ax2.set_xlabel("Missing E [GeV]", fontsize=11)
    ax2.set_ylabel("Events", fontsize=11)
    ax2.set_title("Missing Energy", fontsize=12)
    ax2.grid(alpha=0.3)

    # 3. pT of charged leptons
    ax3 = fig.add_subplot(gs[0, 2])
    for pdg, name, color in [(11, r"$e^{\pm}$", "#1f77b4"),
                              (13, r"$\mu^{\pm}$", "#ff7f0e")]:
        pts = analyzer.pt_spectrum(pdg)
        if len(pts):
            ax3.hist(pts, bins=35, alpha=0.6, label=name, color=color)
    ax3.set_xlabel(r"$p_T$ [GeV]", fontsize=11)
    ax3.set_ylabel("Particles", fontsize=11)
    ax3.set_title("Lepton pT", fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Eta distribution
    ax4 = fig.add_subplot(gs[1, 0])
    for pdg, name, color in [(11, r"$e^{\pm}$", "#1f77b4"),
                              (13, r"$\mu^{\pm}$", "#ff7f0e")]:
        etas = analyzer.eta_distribution(pdg)
        if len(etas):
            ax4.hist(etas, bins=35, alpha=0.6, label=name, color=color)
    ax4.set_xlabel(r"$\eta$", fontsize=11)
    ax4.set_ylabel("Particles", fontsize=11)
    ax4.set_title("Pseudorapidity", fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Angular distribution of leptons
    ax5 = fig.add_subplot(gs[1, 1])
    cos_theta = analyzer.cos_theta_distribution(11)
    if len(cos_theta) == 0:
        cos_theta = analyzer.cos_theta_distribution(13)
    if len(cos_theta):
        plot_angular_distribution(cos_theta, title=r"$\cos\theta$ (leptons)", ax=ax5)

    # 6. Decay mode pie chart
    ax6 = fig.add_subplot(gs[1, 2])
    fracs = analyzer.decay_mode_fractions()
    if fracs:
        labels = [k[:25] for k in fracs.keys()]
        vals = list(fracs.values())
        ax6.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
        ax6.set_title("Decay Modes", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig

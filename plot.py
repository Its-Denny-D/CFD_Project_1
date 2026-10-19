"""Plots for the FTCS/ADI heat conduction project.

Reads the CSVs that the Zig solver writes to out/ and saves figures to figs/.
Every constant (grid size, dt, t, L, W, alpha, k, CONSS) comes from each field file's
`#` header line, so nothing here needs keeping in sync with src/main.zig by hand.

Run:  .venv/bin/python plot.py
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FIGS = ROOT / "figs"

# Figure output: "pdf" for LaTeX, "png" for Word
FIG_FORMAT = "pdf"
FIG_DPI = 300
COL_WIDTH = 3.5  # in, one AIAA column
PAGE_WIDTH = 7.0  # in, full AIAA page width

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ---------------------------------------------------------------------------
# Loaders and helpers
# ---------------------------------------------------------------------------

def parse_value(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_field(path):
    """Return (meta dict, T) where T has shape (JMAX, IMAX), row = j."""
    path = Path(path)
    with path.open() as f:
        header = f.readline()
    meta = {}
    for pair in header.lstrip("#").split():
        key, value = pair.split("=", 1)
        meta[key] = parse_value(value)
    T = np.loadtxt(path, delimiter=",")
    assert T.shape == (meta["JMAX"], meta["IMAX"]), f"{path.name}: shape {T.shape} vs header"
    return meta, T


def load_history(path):
    # deletechars='' keeps the dots in probe names like T_x0.10_y0.05
    return np.genfromtxt(path, delimiter=",", names=True, deletechars="")


def grid(meta):
    x = np.linspace(0.0, meta["L"], meta["IMAX"])
    y = np.linspace(0.0, meta["W"], meta["JMAX"])
    return np.meshgrid(x, y)  # X, Y each (JMAX, IMAX), matching T


def probe_names(hist):
    return [n for n in hist.dtype.names if n.startswith("T_")]


def probe_label(name):
    # Contract with src/main.zig (probes): names must be T_x<x>_y<y>, x and y in meters
    # "T_x0.10_y0.05" -> "(x, y) = (0.10, 0.05) m"
    _, xs, ys = name.split("_")
    return f"(x, y) = ({xs[1:]}, {ys[1:]}) m"


def savefig(fig, name):
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f"{name}.{FIG_FORMAT}"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def ftcs_path(dt, tag):
    # Zig's {d} prints 1.0 as "1" and 0.2 as "0.2"
    dt_s = f"{dt:g}"
    return OUT / f"ftcs_dt{dt_s}_{tag}.csv"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def sanity_checks():
    print("Sanity checks")
    ok = True
    for tag in ["t10", "t40", "steady"]:
        meta, T = load_field(ftcs_path(0.2, tag))
        p = f"ftcs dt=0.2 {tag}"
        ok &= check(f"{p}: bottom row = 40", np.allclose(T[0, 1:-1], 40.0))
        ok &= check(f"{p}: top row = 10", np.allclose(T[-1, 1:-1], 10.0))
        ok &= check(f"{p}: side columns = 0",
                    np.allclose(T[1:-1, 0], 0.0) and np.allclose(T[1:-1, -1], 0.0))
        asym = np.max(np.abs(T - np.fliplr(T)))
        ok &= check(f"{p}: left-right symmetry", asym < 1e-10, f"max |T - fliplr(T)| = {asym:.2e}")

    hist = load_history(ftcs_path(0.2, "history"))
    ok &= check("ftcs dt=0.2: last TV < CONSS", hist["tv"][-1] < meta["CONSS"], f"{hist['tv'][-1]:.4g}")
    ok &= check("ftcs dt=0.2: history starts at t = 0", hist["t"][0] == 0.0)
    return ok


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

LEVELS = np.linspace(0.0, 40.0, 41)


def fig_ftcs_snapshots(dt=0.2):
    """F1: contours at t = 10 s, 40 s and steady state, shared levels and colorbar."""
    tags = ["t10", "t40", "steady"]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.4), sharey=True, constrained_layout=True)
    for ax, tag in zip(axes, tags):
        meta, T = load_field(ftcs_path(dt, tag))
        X, Y = grid(meta)
        cf = ax.contourf(X, Y, T, levels=LEVELS, cmap="inferno")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        title = "steady state" if tag == "steady" else f"t = {meta['t']:g} s"
        ax.set_title(f"{title}" + (f" (t = {meta['t']:g} s)" if tag == "steady" else ""))
    axes[0].set_ylabel("y (m)")
    cb = fig.colorbar(cf, ax=axes, shrink=0.9)
    cb.set_label("T (°C)")
    cb.set_ticks(np.arange(0, 41, 10))
    savefig(fig, f"fig_ftcs_dt{dt:g}_snapshots")


def fig_ftcs_profiles(dt=0.2):
    """F2: T vs y for x = 0, 0.05, ..., 0.30 m, one panel per time (replaces the Part I table)."""
    tags = ["t10", "t40", "steady"]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.8), sharey=True, constrained_layout=True)
    for ax, tag in zip(axes, tags):
        meta, T = load_field(ftcs_path(dt, tag))
        L, W = meta["L"], meta["W"]
        y = np.linspace(0.0, W, meta["JMAX"])
        step = round(0.05 / (L / (meta["IMAX"] - 1)))
        for i in range(0, meta["IMAX"], step):
            x = i * L / (meta["IMAX"] - 1)
            # dashed right half: symmetric pairs (0.05/0.25, ...) overlay, so both stay visible
            ax.plot(y, T[:, i], ls="--" if x > L / 2 + 1e-9 else "-", label=f"x = {x:.2f} m")
        title = "steady state" if tag == "steady" else f"t = {meta['t']:g} s"
        ax.set_title(title)
        ax.set_xlabel("y (m)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("T (°C)")
    axes[-1].legend(loc="upper right", ncol=1)
    savefig(fig, f"fig_ftcs_dt{dt:g}_profiles")


def fig_ftcs_probes(dt=0.2):
    """F3: probe temperatures vs t from t = 0 to steady state."""
    hist = load_history(ftcs_path(dt, "history"))
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6), constrained_layout=True)
    for name in probe_names(hist):
        ax.plot(hist["t"], hist[name], label=probe_label(name))
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, f"fig_ftcs_dt{dt:g}_probes")


def fig_ftcs_convergence(dt=0.2):
    """F4: TV vs t on a log axis; late-time slope vs the slowest decay mode."""
    meta, _ = load_field(ftcs_path(dt, "steady"))
    hist = load_history(ftcs_path(dt, "history"))
    t, tv = hist["t"][1:], hist["tv"][1:]  # step 0 has no TV

    # Fit ln(TV) = a - lambda t over the second half of the run
    late = t > 0.5 * t[-1]
    lam_fit = -np.polyfit(t[late], np.log(tv[late]), 1)[0]
    lam_theory = meta["alpha"] * np.pi**2 * (1.0 / meta["L"] ** 2 + 1.0 / meta["W"] ** 2)
    print(f"  decay rate: fit {lam_fit:.5f} 1/s, theory {lam_theory:.5f} 1/s "
          f"({100 * (lam_fit / lam_theory - 1):+.2f} %)")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6), constrained_layout=True)
    ax.semilogy(t, tv, label="TV (FTCS)")
    ax.axhline(meta["CONSS"], color="k", ls="--", lw=0.8, label=f"CONSS = {meta['CONSS']:g} °C")
    t0 = t[late][0]
    tv0 = tv[late][0]
    ax.semilogy(t[late], tv0 * np.exp(-lam_theory * (t[late] - t0)), ":", color="C3",
                label=rf"slowest mode, $\lambda$ = {lam_theory:.4f} s$^{{-1}}$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("TV (°C)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    savefig(fig, f"fig_ftcs_dt{dt:g}_convergence")


def fig_ftcs_instability(dt=1.0):
    """F5: growth of |T| at a probe vs step, overlaid with the von Neumann growth factor."""
    meta, _ = load_field(ftcs_path(dt, "t0"))
    hist = load_history(ftcs_path(dt, "history"))
    dx = meta["L"] / (meta["IMAX"] - 1)
    dy = meta["W"] / (meta["JMAX"] - 1)
    # Worst (checkerboard) mode: G = 1 - 4 (d_x + d_y)
    G = 1.0 - 4.0 * meta["alpha"] * dt * (1.0 / dx**2 + 1.0 / dy**2)

    name = probe_names(hist)[0]
    n = hist["step"]
    T = hist[name]
    mag = np.abs(T)
    good = np.isfinite(mag) & (mag > 0)

    # Fit the observed growth rate over the last half of the finite data
    ng, mg = n[good], mag[good]
    late = ng > ng[0] + 0.5 * (ng[-1] - ng[0])
    slope_fit = np.polyfit(ng[late], np.log10(mg[late]), 1)[0]
    print(f"  growth: fit {slope_fit:.3f} decades/step, theory log10|G| = {np.log10(abs(G)):.3f} "
          f"(|G| = {abs(G):.3f})")

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6), constrained_layout=True)
    # log10|T| on a linear axis: a log axis spanning ~300 decades overflows f64 in matplotlib,
    # and the slope then reads directly in decades per step
    lg = np.log10(mg)
    n_ref = ng[late][-1]
    ax_log.plot(ng, lg, label=f"|T| at {probe_label(name)}")
    ax_log.plot(ng[late], lg[late][-1] + np.log10(abs(G)) * (ng[late] - n_ref), "--", color="C3",
                label=rf"$|G|^n$, $|G|$ = {abs(G):.2f}")
    ax_log.set_xlabel("step n")
    ax_log.set_ylabel(r"$\log_{10}$ |T| (°C)")
    ax_log.set_title(f"FTCS, Δt = {dt:g} s")
    ax_log.grid(alpha=0.3)
    ax_log.legend()

    first = n <= 40
    ax_lin.plot(n[first], T[first], "o-", ms=2.5)
    # symlog: linear near 0, log beyond, so the sign alternation shows across all magnitudes
    ax_lin.set_yscale("symlog", linthresh=1.0)
    ax_lin.set_xlabel("step n")
    ax_lin.set_ylabel("T (°C)")
    ax_lin.set_title(f"First 40 steps, {probe_label(name)}")
    ax_lin.grid(alpha=0.3)
    savefig(fig, f"fig_ftcs_dt{dt:g}_instability")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ok = sanity_checks()
    if not ok:
        print("Sanity checks failed; figures may be wrong.", file=sys.stderr)

    print("Figures")
    fig_ftcs_snapshots()
    fig_ftcs_probes()
    fig_ftcs_convergence()
    fig_ftcs_instability()
    fig_ftcs_profiles()
    # TODO F6-F9 once analytical.csv and ADI output exist
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

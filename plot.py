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
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FIGS = ROOT / "figs"
ANALYTICAL = OUT / "analytical.csv"

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


# Contract with src/main.zig: every solver writes <method>_dt<dt>_<tag>.csv with the
# same tags (t0, t10, t40, steady, history) and the same history columns.
METHODS = ("ftcs", "adi")
METHOD_LABEL = {"ftcs": "FTCS", "adi": "ADI"}


def field_path(method, dt, tag):
    # Zig's {d} prints 1.0 as "1" and 0.2 as "0.2"
    return OUT / f"{method}_dt{dt:g}_{tag}.csv"


def label(method):
    return METHOD_LABEL.get(method, method.upper())


def err_label(method):
    return rf"$T_{{\mathrm{{{label(method)}}}}} - T_{{\mathrm{{exact}}}}$ (°C)"


def fig_name(method, dt, name):
    return f"fig_{method}_dt{dt:g}_{name}"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def sanity_checks(method="ftcs", dt=0.2):
    print(f"Sanity checks, {label(method)}")
    ok = True
    for tag in ["t10", "t40", "steady"]:
        meta, T = load_field(field_path(method, dt, tag))
        p = f"{method} dt={dt:g} {tag}"
        ok &= check(f"{p}: bottom row = 40", np.allclose(T[0, 1:-1], 40.0))
        ok &= check(f"{p}: top row = 10", np.allclose(T[-1, 1:-1], 10.0))
        ok &= check(f"{p}: side columns = 0",
                    np.allclose(T[1:-1, 0], 0.0) and np.allclose(T[1:-1, -1], 0.0))
        asym = np.max(np.abs(T - np.fliplr(T)))
        ok &= check(f"{p}: left-right symmetry", asym < 1e-10, f"max |T - fliplr(T)| = {asym:.2e}")

    # CONSS from the steady file that belongs to this history, not whichever file loaded last
    conss = load_field(field_path(method, dt, "steady"))[0]["CONSS"]
    hist = load_history(field_path(method, dt, "history"))
    ok &= check(f"{method} dt={dt:g}: last TV < CONSS", hist["tv"][-1] < conss, f"{hist['tv'][-1]:.4g}")
    ok &= check(f"{method} dt={dt:g}: history starts at t = 0", hist["t"][0] == 0.0)
    return ok


def load_pair(method="ftcs", dt=0.2):
    """Analytical and numerical steady fields on the same grid: (meta, T_exact, T_num)."""
    meta_a, T_a = load_field(ANALYTICAL)
    meta_f, T_f = load_field(field_path(method, dt, "steady"))
    # a mismatch would make the error field meaningless
    for key in ("IMAX", "JMAX", "L", "W"):
        assert meta_a[key] == meta_f[key], f"analytical and {label(method)} headers disagree on {key}"
    return meta_f, T_a, T_f


# Interior-only error metrics: the walls are identical by construction, so including
# them would dilute the averages. Absolute error in C, not relative: T -> 0 near the
# side walls, so a relative error blows up there.
NEAR_HOT = 2  # interior rows / columns excluded from the "bulk" metric


def interior_error(T_exact, T_num):
    return (T_num - T_exact)[1:-1, 1:-1]  # e = numerical - analytical, indices shifted by 1


def analytical_checks(method="ftcs", dt=0.2):
    print(f"Analytical sanity checks (against {label(method)})")
    ok = True
    meta, T_a, T_f = load_pair(method, dt)
    ok &= check("analytical: bottom row = 40", np.allclose(T_a[0, 1:-1], 40.0))
    ok &= check("analytical: top row = 10", np.allclose(T_a[-1, 1:-1], 10.0))
    ok &= check("analytical: side columns = 0",
                np.allclose(T_a[1:-1, 0], 0.0) and np.allclose(T_a[1:-1, -1], 0.0))
    ok &= check(f"analytical: walls and corners match the {label(method)} field",
                np.array_equal(T_a[0], T_f[0]) and np.array_equal(T_a[-1], T_f[-1])
                and np.array_equal(T_a[:, 0], T_f[:, 0]) and np.array_equal(T_a[:, -1], T_f[:, -1]))
    asym = np.max(np.abs(T_a - np.fliplr(T_a)))
    ok &= check("analytical: left-right symmetry", asym < 1e-10, f"{asym:.2e}")
    jc, ic = meta["JMAX"] // 2, meta["IMAX"] // 2
    ok &= check("analytical: center node near 7.68 (this problem's value)",
                abs(T_a[jc, ic] - 7.68) < 0.01, f"T[{jc},{ic}] = {T_a[jc, ic]:.4f}")
    e = interior_error(T_a, T_f)
    ok &= check(f"smoke test: max |{label(method)} - analytical| < 0.5 C", np.max(np.abs(e)) < 0.5,
                f"{np.max(np.abs(e)):.3f}")
    return ok


def error_metrics(method="ftcs", dt=0.2):
    """Console metrics for a steady field vs the analytical field, interior nodes only."""
    meta, T_a, T_f = load_pair(method, dt)
    e = interior_error(T_a, T_f)
    x = np.linspace(0.0, meta["L"], meta["IMAX"])[1:-1]
    y = np.linspace(0.0, meta["W"], meta["JMAX"])[1:-1]

    j, i = np.unravel_index(np.argmax(np.abs(e)), e.shape)
    lab = label(method)
    print(f"Error metrics, {lab} dt={dt:g} steady vs analytical (interior, e = {lab} - exact)")
    print(f"  max |e|       = {abs(e[j, i]):.4f} C at (x, y) = ({x[i]:.3f}, {y[j]:.3f}) m, signed {e[j, i]:+.4f}")
    print(f"  RMS e         = {np.sqrt(np.mean(e**2)):.4f} C")
    print(f"  mean e        = {np.mean(e):+.4f} C")
    # About -0.006 if the stopping error dominates at the center (cheap check before F6c)
    jc, ic = meta["JMAX"] // 2 - 1, meta["IMAX"] // 2 - 1  # interior indices
    print(f"  center e      = {e[jc, ic]:+.4f} C at (x, y) = ({x[ic]:.3f}, {y[jc]:.3f}) m")

    # Bulk: drop the first NEAR_HOT interior rows from the hot wall and NEAR_HOT interior
    # columns beside each side wall, to separate the hot (bottom) corner singularity from the
    # bulk error. The top corners (10 -> 0 C) also jump, but much less, and stay included.
    mask = np.ones_like(e, dtype=bool)
    mask[:NEAR_HOT, :] = False
    mask[:, :NEAR_HOT] = False
    mask[:, -NEAR_HOT:] = False
    jb, ib = np.unravel_index(np.argmax(np.where(mask, np.abs(e), -1.0)), e.shape)
    print(f"  max |e| bulk  = {abs(e[jb, ib]):.4f} C at (x, y) = ({x[ib]:.3f}, {y[jb]:.3f}) m "
          f"(excluding {NEAR_HOT} rows / columns near the hot (bottom) corners)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

LEVELS = np.linspace(0.0, 40.0, 41)


def fig_snapshots(method="ftcs", dt=0.2):
    """F1: contours at t = 10 s, 40 s and steady state, shared levels and colorbar."""
    tags = ["t10", "t40", "steady"]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.4), sharey=True, constrained_layout=True)
    for ax, tag in zip(axes, tags):
        meta, T = load_field(field_path(method, dt, tag))
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
    savefig(fig, fig_name(method, dt, "snapshots"))


def fig_profiles(method="ftcs", dt=0.2):
    """F2: T vs y for x = 0, 0.05, ..., 0.30 m, one panel per time (replaces the Part I table)."""
    tags = ["t10", "t40", "steady"]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.8), sharey=True, constrained_layout=True)
    for ax, tag in zip(axes, tags):
        meta, T = load_field(field_path(method, dt, tag))
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
    savefig(fig, fig_name(method, dt, "profiles"))


def fig_probes(method="ftcs", dt=0.2):
    """F3: probe temperatures vs t from t = 0 to steady state."""
    hist = load_history(field_path(method, dt, "history"))
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6), constrained_layout=True)
    for name in probe_names(hist):
        ax.plot(hist["t"], hist[name], label=probe_label(name))
    ax.set_xlabel("t (s)")
    ax.set_ylabel("T (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, fig_name(method, dt, "probes"))


def fig_convergence(method="ftcs", dt=0.2):
    """F4: TV vs t on a log axis; late-time slope vs the slowest decay mode."""
    meta, _ = load_field(field_path(method, dt, "steady"))
    hist = load_history(field_path(method, dt, "history"))
    t, tv = hist["t"][1:], hist["tv"][1:]  # step 0 has no TV

    # Fit ln(TV) = a - lambda t over the second half of the run
    late = t > 0.5 * t[-1]
    lam_fit = -np.polyfit(t[late], np.log(tv[late]), 1)[0]
    lam_theory = meta["alpha"] * np.pi**2 * (1.0 / meta["L"] ** 2 + 1.0 / meta["W"] ** 2)
    print(f"  decay rate: fit {lam_fit:.5f} 1/s, theory {lam_theory:.5f} 1/s "
          f"({100 * (lam_fit / lam_theory - 1):+.2f} %)")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6), constrained_layout=True)
    ax.semilogy(t, tv, label=f"TV ({label(method)})")
    ax.axhline(meta["CONSS"], color="k", ls="--", lw=0.8, label=f"CONSS = {meta['CONSS']:g} °C")
    t0 = t[late][0]
    tv0 = tv[late][0]
    ax.semilogy(t[late], tv0 * np.exp(-lam_theory * (t[late] - t0)), ":", color="C3",
                label=rf"slowest mode, $\lambda$ = {lam_theory:.4f} s$^{{-1}}$")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("TV (°C)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    savefig(fig, fig_name(method, dt, "convergence"))


def fig_ftcs_instability(dt=1.0):
    """F5: growth of |T| at a probe vs step, overlaid with the von Neumann growth factor.

    FTCS only: G is the FTCS amplification factor (ADI is unconditionally stable)."""
    meta, _ = load_field(field_path("ftcs", dt, "t0"))
    hist = load_history(field_path("ftcs", dt, "history"))
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
    savefig(fig, fig_name("ftcs", dt, "instability"))


def fig_centerline(method="ftcs", dt=0.2):
    """F7: T(y) at x = L/2, analytical line + numerical markers, error on a second axis."""
    meta, T_a, T_f = load_pair(method, dt)
    i = meta["IMAX"] // 2
    x = i * meta["L"] / (meta["IMAX"] - 1)
    y = np.linspace(0.0, meta["W"], meta["JMAX"])
    err = T_f[:, i] - T_a[:, i]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.8), constrained_layout=True)
    ax.plot(y, T_a[:, i], "-", color="C0", label="analytical")
    ax.plot(y[::2], T_f[::2, i], "o", color="C1", ms=3, mfc="none", label=label(method))
    ax.set_xlabel("y (m)")
    ax.set_ylabel("T (°C)")
    ax.set_title(f"Centerline x = {x:.2f} m")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(y, err, "-", color="C3", lw=0.9, label="error")
    ax2.axhline(0.0, color="C3", lw=0.5, ls=":")
    ax2.set_ylabel(err_label(method), color="C3")
    ax2.tick_params(axis="y", colors="C3")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # below the axes: every spot inside collides with the temperature or the error curve
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=3)
    savefig(fig, fig_name(method, dt, "centerline"))


def fig_steady_comparison(method="ftcs", dt=0.2):
    """F6: analytical, numerical steady and the error field."""
    meta, T_a, T_f = load_pair(method, dt)
    X, Y = grid(meta)
    err = T_f - T_a

    # Symmetric limits clipped at the 99th percentile of |e|: the corners would
    # otherwise dominate the colour scale. extend="both" paints what exceeds it.
    interior = np.abs(err[1:-1, 1:-1])
    lim = float(np.percentile(interior, 99))
    jm, im = np.unravel_index(np.argmax(interior), interior.shape)
    jm, im = jm + 1, im + 1

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.4), sharey=True, constrained_layout=True)
    cf = None
    for ax, T, title in zip(axes[:2], (T_a, T_f), ("analytical", f"{label(method)} steady")):
        cf = ax.contourf(X, Y, T, levels=LEVELS, cmap="inferno")
        ax.set_title(title)
    cb = fig.colorbar(cf, ax=axes[:2], shrink=0.8, location="bottom", pad=0.02)
    cb.set_label("T (°C)")
    cb.set_ticks(np.arange(0, 41, 10))

    levels_e = np.linspace(-lim, lim, 21)
    ce = axes[2].contourf(X, Y, err, levels=levels_e, cmap="RdBu_r",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), extend="both")
    axes[2].plot(X[jm, im], Y[jm, im], "k+", ms=7, mew=1.2)
    axes[2].annotate(f"max |e| = {interior[jm - 1, im - 1]:.2f} °C", (X[jm, im], Y[jm, im]),
                     xytext=(0.02, 0.2), textcoords="axes fraction", fontsize=7,
                     arrowprops=dict(arrowstyle="-", lw=0.6))
    axes[2].set_title(f"error, {label(method)} − exact")
    cbe = fig.colorbar(ce, ax=axes[2], shrink=0.8, location="bottom", pad=0.02)
    cbe.set_label(err_label(method))
    ticks = np.linspace(-lim, lim, 5)
    cbe.set_ticks(ticks)
    cbe.set_ticklabels([f"{t:.2f}" for t in ticks])
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    savefig(fig, fig_name(method, dt, "steady_comparison"))


def fig_near_wall_error(method="ftcs", dt=0.2):
    """F7b: error along the first interior row (j = 1) vs x."""
    meta, T_a, T_f = load_pair(method, dt)
    x = np.linspace(0.0, meta["L"], meta["IMAX"])
    y1 = meta["W"] / (meta["JMAX"] - 1)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4), constrained_layout=True)
    ax.plot(x[1:-1], (T_f - T_a)[1, 1:-1], "o-", ms=2.5)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel(err_label(method))
    ax.set_title(f"Error along y = {y1:.2f} m")
    ax.grid(alpha=0.3)
    savefig(fig, fig_name(method, dt, "near_wall_error"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dt=0.2):
    # Every method whose steady file exists gets the full set of checks and figures
    methods = [m for m in METHODS if field_path(m, dt, "steady").exists()]
    have_analytical = ANALYTICAL.exists()
    if not have_analytical:
        print(f"{ANALYTICAL.name} not found; skipping analytical checks and figures", file=sys.stderr)

    ok = True
    for method in methods:
        ok &= sanity_checks(method, dt)
        if have_analytical:
            ok &= analytical_checks(method, dt)
            error_metrics(method, dt)
    if not ok:
        print("Sanity checks failed; figures may be wrong.", file=sys.stderr)

    print("Figures")
    for method in methods:
        fig_snapshots(method, dt)
        fig_probes(method, dt)
        fig_convergence(method, dt)
        fig_profiles(method, dt)
        if have_analytical:
            fig_centerline(method, dt)
            fig_steady_comparison(method, dt)
            fig_near_wall_error(method, dt)
    fig_ftcs_instability()
    # TODO F6c (needs a CONSS = 0.001 run), F8 heat flux and the 3.8 FTCS/ADI overlays
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

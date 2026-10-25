"""F8: wall heat flux from the steady fields (Part IV).

Sign convention: q_in = q . n_in with q = -k grad T, so positive means heat entering the bar
through that wall. The gradient at each wall is the second-order one-sided difference, applied
the same way to every field (FTCS, ADI and analytical.csv). Comparing against the analytical field
therefore isolates the error of the numerical solution: both share the truncation error of the
difference formula itself.

Corners are excluded. The wall temperature jumps there, so the exact flux is unbounded and the
spikes next to the corners grow with grid refinement.

Run on its own:  .venv/bin/python heat_flux.py   (plot.py also calls it from main)
"""
import numpy as np
import matplotlib.pyplot as plt

from plot import (load_field, field_path, ANALYTICAL, OUT, METHODS, savefig, label,
                  PAGE_WIDTH)

WALLS = ("bottom", "top", "left", "right")
WALL_TITLE = {
    "bottom": "Bottom wall, y = 0 (40 °C)",
    "top": "Top wall, y = W (10 °C)",
    "left": "Left wall, x = 0 (0 °C)",
    "right": "Right wall, x = L (0 °C)",
}
NEAR_CORNER = 2  # nodes at each end of a wall left out of the "away from corners" error


def wall_flux(T, meta):
    """Heat flow into the bar (W/m^2) along each wall, corners excluded.

    Returns {wall: (s, q)}: s is the coordinate along the wall (m), q the inward flux.
    """
    dx = meta["L"] / (meta["IMAX"] - 1)
    dy = meta["W"] / (meta["JMAX"] - 1)
    k = meta["k"]
    x = np.linspace(0.0, meta["L"], meta["IMAX"])
    y = np.linspace(0.0, meta["W"], meta["JMAX"])

    # T[j, i]: T[0, :] is the bottom row, T[:, 0] the left column, negative indices count from the end
    dTdy_bottom = (-3 * T[0, :] + 4 * T[1, :] - T[2, :]) / (2 * dy)      # forward difference
    dTdy_top = (3 * T[-1, :] - 4 * T[-2, :] + T[-3, :]) / (2 * dy)       # backward difference
    dTdx_left = (-3 * T[:, 0] + 4 * T[:, 1] - T[:, 2]) / (2 * dx)        # forward difference
    dTdx_right = (3 * T[:, -1] - 4 * T[:, -2] + T[:, -3]) / (2 * dx)     # backward difference

    # Inward normals: +y at the bottom, -y at the top, +x on the left, -x on the right
    return {
        "bottom": (x[1:-1], -k * dTdy_bottom[1:-1]),
        "top": (x[1:-1], k * dTdy_top[1:-1]),
        "left": (y[1:-1], -k * dTdx_left[1:-1]),
        "right": (y[1:-1], k * dTdx_right[1:-1]),
    }


def wall_totals(flux):
    """Integrated inward heat flow per unit depth (W/m) on each wall, trapezoid rule."""
    return {w: np.trapezoid(q, s) for w, (s, q) in flux.items()}


def load_fluxes(methods, dt):
    """{name: flux dict} for the analytical field and each numerical method."""
    meta, T_exact = load_field(ANALYTICAL)
    fluxes = {"exact": wall_flux(T_exact, meta)}
    for method in methods:
        meta_m, T = load_field(field_path(method, dt, "steady"))
        fluxes[method] = wall_flux(T, meta_m)
    return fluxes


def report(fluxes, methods, dt):
    """Console printout (stands in for the Part IV table) plus the error and balance checks."""
    exact = fluxes["exact"]
    print(f"Wall heat flux q_in (W/m^2, positive = into the bar), steady state, dt={dt:g}")
    for wall in ("bottom", "top", "left"):  # right mirrors left
        s, q_e = exact[wall]
        every = round(0.05 / (s[1] - s[0]))  # 0.05 m spacing, as in Part I
        idx = np.arange(every - 1, len(s), every)
        print(f"  {WALL_TITLE[wall]}")
        print("    " + "s (m)".rjust(8) + "".join(n.rjust(12) for n in ["exact", *(label(m) for m in methods)]))
        for i in idx:
            row = [q_e[i], *(fluxes[m][wall][1][i] for m in methods)]
            print("    " + f"{s[i]:8.2f}" + "".join(f"{v:12.1f}" for v in row))

    print("  Error vs analytical field, away from the corners "
          f"(dropping {NEAR_CORNER} nodes at each end):")
    for m in methods:
        parts = []
        for wall in ("bottom", "top", "left"):
            diff = fluxes[m][wall][1] - exact[wall][1]
            parts.append(f"{wall} {np.max(np.abs(diff[NEAR_CORNER:-NEAR_CORNER])):.2f}")
        print(f"    {label(m)}: max |q - q_exact| = " + ", ".join(parts) + " W/m^2")

    # Steady state: heat in = heat out, so the four integrals should sum to ~0. Only approximate:
    # the corner singularities make the integrated flux near each corner grid-dependent.
    print("  Energy balance, integrated q_in per unit depth (W/m), corners excluded:")
    for name in ("exact", *methods):
        tot = wall_totals(fluxes[name])
        net = sum(tot.values())
        print(f"    {('exact' if name == 'exact' else label(name)):>6}: "
              + ", ".join(f"{w} {tot[w]:+8.1f}" for w in WALLS)
              + f"  | net {net:+7.1f} ({100 * net / tot['bottom']:+.1f} % of heat in)")


def write_csv(fluxes, methods, dt):
    """All wall fluxes in one long-format CSV: wall, s, q_exact, q_<method>..."""
    path = OUT / f"heat_flux_dt{dt:g}.csv"
    with path.open("w") as f:
        f.write(f"# q_in in W/m^2, positive = into the bar; steady fields, dt={dt:g}; corners excluded\n")
        f.write("wall,s," + ",".join(["q_exact", *(f"q_{m}" for m in methods)]) + "\n")
        for wall in WALLS:
            s, q_e = fluxes["exact"][wall]
            for i in range(len(s)):
                vals = [q_e[i], *(fluxes[m][wall][1][i] for m in methods)]
                f.write(f"{wall},{float(s[i])!r}," + ",".join(repr(float(v)) for v in vals) + "\n")
    print(f"  wrote {path}")


def fig_heat_flux(methods, dt=0.2):
    """F8: inward flux along the bottom, top and left walls (top row), error vs analytical (bottom row).

    The right wall mirrors the left, so it is in the CSV and the balance but not plotted.
    """
    fluxes = load_fluxes(methods, dt)
    report(fluxes, methods, dt)
    write_csv(fluxes, methods, dt)

    walls = ("bottom", "top", "left")
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 4.4), sharex="col", constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    styles = {"ftcs": dict(ls="-", color="C0"), "adi": dict(ls="none", marker="o", ms=3, mfc="none", color="C1")}
    for col, wall in enumerate(walls):
        ax, ax_e = axes[0, col], axes[1, col]
        # plotted in kW/m^2 so every panel reads without an axis offset; CSV and console stay in W/m^2
        s, q_e = fluxes["exact"][wall]
        for m in methods:
            ax.plot(s, fluxes[m][wall][1] / 1e3, label=label(m), **styles.get(m, {}))
            ax_e.plot(s, (fluxes[m][wall][1] - q_e) / 1e3, label=label(m), **styles.get(m, {}))
        ax.plot(s, q_e / 1e3, "--", color="k", lw=0.9, label="analytical field")
        ax.axhline(0.0, color="k", lw=0.4)
        ax_e.axhline(0.0, color="k", lw=0.4)
        ax.set_title(WALL_TITLE[wall])
        ax.grid(alpha=0.3)
        ax_e.grid(alpha=0.3)
        ax_e.set_xlabel("y (m)" if wall == "left" else "x (m)")
    axes[0, 0].set_ylabel(r"$q_{\mathrm{in}}$ (kW/m$^2$)")
    axes[1, 0].set_ylabel(r"$q - q_{\mathrm{exact}}$ (kW/m$^2$)")
    axes[0, 0].legend(loc="upper center")
    savefig(fig, f"fig_dt{dt:g}_heat_flux")


if __name__ == "__main__":
    fig_heat_flux([m for m in METHODS if field_path(m, 0.2, "steady").exists()])

const std = @import("std");
const Io = std.Io;

fn idx(i: usize, j: usize) usize {
    return j * IMAX + i;
}
// Step Bounds
const IMAX: usize = 31;
const JMAX: usize = 41;

// Physical parameters
const L: f64 = 0.3; //meters
const W: f64 = 0.4; //meters
const Thermal_diffusivity: f64 = 11.234e-5; //m^2/s
const Thermal_conductivity: f64 = 380; //W/mC

//step sizes
const dx: f64 = L / @as(f64, @floatFromInt(IMAX - 1));
const dy: f64 = W / @as(f64, @floatFromInt(JMAX - 1));

// Boundary conditions
const T_0: f64 = 0.0; //C
const T_1: f64 = 40.0; //C
const T_2: f64 = 0.0; //C
const T_3: f64 = 10.0; //C
const T_4: f64 = 0.0; //C

// steady state solution
const t_0_steady: f64 = 10.0; //s
const t_1_steady: f64 = 40.0; //s
const interval_steady: f64 = 0.05; //m

// Exit Conditions
const CONSS: f64 = 0.01; // C

// Output
const OUT_DIR = "out";

// History probes; each adds a column to the history CSV.
// Contract with plot.py (probe_label): names must be T_x<x>_y<y>, x and y in meters.
const Probe = struct { name: []const u8, i: usize, j: usize };
const probes = [_]Probe{
    .{ .name = "T_x0.10_y0.05", .i = 10, .j = 5 },
    .{ .name = "T_x0.15_y0.10", .i = 15, .j = 10 },
    .{ .name = "T_x0.10_y0.30", .i = 10, .j = 30 },
};

fn print_grid(grid: []const f64) void {
    // Rows are j (top boundary last), columns are i; every 5th node, scientific notation
    var j: usize = 0;
    while (j < JMAX) : (j += 5) {
        var i: usize = 0;
        while (i < IMAX) : (i += 5) {
            std.debug.print("{e:10.2}", .{grid[idx(i, j)]});
        }
        std.debug.print("\n", .{});
    }
}

const FieldMeta = struct { method: []const u8, dt: f64, t: f64 };

// One field as CSV: a `#` metadata line, then JMAX rows (j = 0 first) of IMAX values
fn write_field(io: Io, path: []const u8, field: []const f64, meta: FieldMeta) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;

    // plot.py reads every constant it needs from this line
    try w.print("# method={s} dt={d} t={d} IMAX={d} JMAX={d} L={d} W={d} alpha={e} k={d} CONSS={d}\n", .{
        meta.method, meta.dt, meta.t,              IMAX,                 JMAX,
        L,           W,       Thermal_diffusivity, Thermal_conductivity, CONSS,
    });
    for (0..JMAX) |j| {
        for (0..IMAX) |i| {
            if (i != 0) try w.writeAll(",");
            // no precision: shortest text that round-trips to the same f64
            try w.print("{e}", .{field[idx(i, j)]});
        }
        try w.writeAll("\n");
    }
    try w.flush();
}

const Status = enum { converged, diverged, step_limit };

const FtcsResult = struct {
    field: []f64, // newest field, always the buffer `current` ended on
    steps: usize,
    status: Status,
};

// Explicit FTCS solver for one time step size dt
fn ftcs_solve(io: Io, dt: f64, nodes: []f64, nodes_swap: []f64) !FtcsResult {
    // Fourier numbers, computed once before the loop
    const dx_coef: f64 = Thermal_diffusivity * dt / (dx * dx);
    const dy_coef: f64 = Thermal_diffusivity * dt / (dy * dy);

    // Boundaries live in both buffers so the swap never reads stale edges
    @memset(nodes, T_0);
    for (0..IMAX) |i| {
        nodes[idx(i, JMAX - 1)] = T_3; // top
        nodes[idx(i, 0)] = T_1; // bottom
    }
    for (0..JMAX) |j| {
        nodes[idx(0, j)] = T_2; // left
        nodes[idx(IMAX - 1, j)] = T_4; // right
    }
    // Corners: average of the two adjacent walls (not used by the stencil)
    nodes[idx(0, 0)] = (T_1 + T_2) / 2.0; // bottom-left
    nodes[idx(IMAX - 1, 0)] = (T_1 + T_4) / 2.0; // bottom-right
    nodes[idx(0, JMAX - 1)] = (T_3 + T_2) / 2.0; // top-left
    nodes[idx(IMAX - 1, JMAX - 1)] = (T_3 + T_4) / 2.0; // top-right
    @memcpy(nodes_swap, nodes);

    std.debug.print("=== FTCS dt = {d} s ===\n", .{dt});
    std.debug.print("Stability: d_x + d_y = {d:.3} (limit 0.5)\n", .{dx_coef + dy_coef});
    print_grid(nodes);

    var path_buf: [128]u8 = undefined;
    try write_field(io, try std.fmt.bufPrint(&path_buf, OUT_DIR ++ "/ftcs_dt{d}_t0.csv", .{dt}), nodes, .{ .method = "ftcs", .dt = dt, .t = 0.0 });

    // Snapshot steps from snapshot times; integer compare, never step * dt == t
    const snap_times = [_]f64{ t_0_steady, t_1_steady };
    var snap_steps: [snap_times.len]usize = undefined;
    for (snap_times, &snap_steps) |t, *s| s.* = @intFromFloat(@round(t / dt));

    // History file stays open for the whole run
    const hist_file = try Io.Dir.cwd().createFile(io, try std.fmt.bufPrint(&path_buf, OUT_DIR ++ "/ftcs_dt{d}_history.csv", .{dt}), .{});
    defer hist_file.close(io);
    var hist_buf: [4096]u8 = undefined;
    var hist_fw = hist_file.writer(io, &hist_buf);
    const hist = &hist_fw.interface;
    try hist.writeAll("step,t");
    for (probes) |p| try hist.print(",{s}", .{p.name});
    try hist.writeAll(",tv\n");
    // Step 0: initial condition, empty tv (reads as NaN)
    try hist.writeAll("0,0e0");
    for (probes) |p| try hist.print(",{e}", .{nodes[idx(p.i, p.j)]});
    try hist.writeAll(",\n");

    var current = nodes;
    var next = nodes_swap;
    var step: usize = 0;
    const step_limit: usize = 10000;
    const status: Status = while (true) {
        var tv: f64 = 0.0;
        for (1..JMAX - 1) |j| {
            for (1..IMAX - 1) |i| {
                const k = idx(i, j);
                next[k] = current[k] +
                    dx_coef * (current[idx(i + 1, j)] - 2.0 * current[k] + current[idx(i - 1, j)]) +
                    dy_coef * (current[idx(i, j + 1)] - 2.0 * current[k] + current[idx(i, j - 1)]);
                tv += @abs(next[k] - current[k]);
            }
        }

        // swap so `current` always holds the newest step
        std.mem.swap([]f64, &current, &next);
        step += 1;
        const t = @as(f64, @floatFromInt(step)) * dt;

        try hist.print("{d},{e}", .{ step, t });
        for (probes) |p| try hist.print(",{e}", .{current[idx(p.i, p.j)]});
        try hist.print(",{e}\n", .{tv});

        // Name and tag snapshots with the exact snapshot time, not step * dt
        for (snap_steps, snap_times) |s, snap_t| {
            if (step == s) {
                const path = try std.fmt.bufPrint(&path_buf, OUT_DIR ++ "/ftcs_dt{d}_t{d}.csv", .{ dt, snap_t });
                try write_field(io, path, current, .{ .method = "ftcs", .dt = dt, .t = snap_t });
            }
        }

        if (!std.math.isFinite(tv)) break .diverged;
        if (tv < CONSS) break .converged;
        if (step == step_limit) break .step_limit;
    };
    try hist.flush();

    const t_end = @as(f64, @floatFromInt(step)) * dt;
    switch (status) {
        .converged => std.debug.print("Converged after {d} steps (t = {d} s)\n", .{ step, t_end }),
        .diverged => std.debug.print("Diverged after {d} steps (t = {d} s)\n", .{ step, t_end }),
        .step_limit => std.debug.print("Hit step limit of {d} steps (t = {d} s)\n", .{ step_limit, t_end }),
    }

    // End field: ftcs_dt0.2_steady.csv when converged, otherwise named by status
    const end_tag = if (status == .converged) "steady" else @tagName(status);
    const end_path = try std.fmt.bufPrint(&path_buf, OUT_DIR ++ "/ftcs_dt{d}_{s}.csv", .{ dt, end_tag });
    try write_field(io, end_path, current, .{ .method = "ftcs", .dt = dt, .t = t_end });

    return .{ .field = current, .steps = step, .status = status };
}

pub fn main(init: std.process.Init) !void {
    // Print constants
    std.debug.print(
        \\=== Constants ===
        \\Grid:          IMAX = {d}, JMAX = {d}
        \\Domain:        L = {d} m, W = {d} m
        \\Diffusivity:   Thermal_diffusivity = {e} m^2/s
        \\Conductivity:  Thermal_conductivity = {d} W/mC
        \\Step sizes:    dx = {d}, dy = {d}
        \\Boundaries:    T_0 = {d}, T_1 = {d}, T_2 = {d}, T_3 = {d}, T_4 = {d} C
        \\Steady state:  t_0 = {d} s, t_1 = {d} s, interval = {d} m
        \\Exit:          CONSS = {d} C
        \\
    , .{
        IMAX,                JMAX,
        L,                   W,
        Thermal_diffusivity, Thermal_conductivity,
        dx,                  dy,
        T_0,                 T_1,
        T_2,                 T_3,
        T_4,                 t_0_steady,
        t_1_steady,          interval_steady,
        CONSS,
    });

    // initialize allocators and io
    const allocator = init.arena.allocator();
    const io = init.io;
    try Io.Dir.cwd().createDirPath(io, OUT_DIR);

    // FTCS INIT
    const ftcs_step_sz = [_]f64{ 0.2, 1.0 }; //s
    const ftcs_nodes: []f64 = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for ftcs_nodes: {}\n", .{err});
        return err;
    }; //m
    const ftcs_nodes_swap: []f64 = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for ftcs_nodes_swap: {}\n", .{err});
        return err;
    }; //m

    for (ftcs_step_sz) |dt| {
        const result = try ftcs_solve(io, dt, ftcs_nodes, ftcs_nodes_swap);
        print_grid(result.field);
    }

    // ADI INIT

    // Run methods

    // Print To File

}

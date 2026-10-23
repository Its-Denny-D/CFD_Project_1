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

// Walls and corners of a field, shared by every solver so boundaries are set one way.
// Leaves the interior untouched.
fn set_boundaries(field: []f64) void {
    for (0..IMAX) |i| {
        field[idx(i, JMAX - 1)] = T_3; // top
        field[idx(i, 0)] = T_1; // bottom
    }
    for (0..JMAX) |j| {
        field[idx(0, j)] = T_2; // left
        field[idx(IMAX - 1, j)] = T_4; // right
    }
    // Corners: average of the two adjacent walls (not used by the stencil)
    field[idx(0, 0)] = (T_1 + T_2) / 2.0; // bottom-left
    field[idx(IMAX - 1, 0)] = (T_1 + T_4) / 2.0; // bottom-right
    field[idx(0, JMAX - 1)] = (T_3 + T_2) / 2.0; // top-left
    field[idx(IMAX - 1, JMAX - 1)] = (T_3 + T_4) / 2.0; // top-right
}

const Status = enum { converged, diverged, step_limit };

const SolveResult = struct {
    field: []f64, // newest field, always the buffer `current` ended on
    steps: usize,
    status: Status,
};

const STEP_LIMIT: usize = 10000;
const snap_times = [_]f64{ t_0_steady, t_1_steady };

// Files every solver writes, identical in format so plot.py treats FTCS and ADI alike:
//   <method>_dt<dt>_t0.csv, _t10.csv / _t40.csv snapshots, _history.csv, and the end field
//   (_steady.csv when converged, otherwise named by status).
// Initialize in place with open(): hist_fw points into hist_buf, so the struct must not be
// copied or moved afterwards.
const RunOutput = struct {
    io: Io,
    method: []const u8,
    dt: f64,
    snap_steps: [snap_times.len]usize,
    path_buf: [128]u8,
    hist_file: Io.File,
    hist_buf: [4096]u8,
    hist_fw: Io.File.Writer,

    // Writes the t = 0 field and opens the history file with its header and step-0 row
    fn open(self: *RunOutput, io: Io, method: []const u8, dt: f64, initial: []const f64) !void {
        self.io = io;
        self.method = method;
        self.dt = dt;
        try write_field(io, try self.path("t0"), initial, .{ .method = method, .dt = dt, .t = 0.0 });

        // Snapshot steps from snapshot times; integer compare, never step * dt == t
        for (snap_times, &self.snap_steps) |t, *s| s.* = @intFromFloat(@round(t / dt));

        self.hist_file = try Io.Dir.cwd().createFile(io, try self.path("history"), .{});
        errdefer self.hist_file.close(io);
        self.hist_fw = self.hist_file.writer(io, &self.hist_buf);
        const hist = &self.hist_fw.interface;
        try hist.writeAll("step,t");
        for (probes) |p| try hist.print(",{s}", .{p.name});
        try hist.writeAll(",tv\n");
        // Step 0: initial condition, empty tv (reads as NaN)
        try hist.writeAll("0,0e0");
        for (probes) |p| try hist.print(",{e}", .{initial[idx(p.i, p.j)]});
        try hist.writeAll(",\n");
    }

    fn close(self: *RunOutput) void {
        self.hist_file.close(self.io);
    }

    // OUT_DIR/<method>_dt<dt>_<tag>.csv, valid until the next call
    fn path(self: *RunOutput, tag: []const u8) ![]u8 {
        return std.fmt.bufPrint(&self.path_buf, OUT_DIR ++ "/{s}_dt{d}_{s}.csv", .{ self.method, self.dt, tag });
    }

    // One history row per step, plus a snapshot file when step hits a snapshot time
    fn logStep(self: *RunOutput, step: usize, field: []const f64, tv: f64) !void {
        const t = @as(f64, @floatFromInt(step)) * self.dt;
        const hist = &self.hist_fw.interface;
        try hist.print("{d},{e}", .{ step, t });
        for (probes) |p| try hist.print(",{e}", .{field[idx(p.i, p.j)]});
        try hist.print(",{e}\n", .{tv});

        // Name and tag snapshots with the exact snapshot time, not step * dt
        for (self.snap_steps, snap_times) |s, snap_t| {
            if (step == s) {
                var tag_buf: [32]u8 = undefined;
                const tag = try std.fmt.bufPrint(&tag_buf, "t{d}", .{snap_t});
                try write_field(self.io, try self.path(tag), field, .{ .method = self.method, .dt = self.dt, .t = snap_t });
            }
        }
    }

    // Flushes the history, reports how the run ended and writes the end field
    fn finish(self: *RunOutput, status: Status, step: usize, field: []const f64) !void {
        try self.hist_fw.interface.flush();

        const t_end = @as(f64, @floatFromInt(step)) * self.dt;
        switch (status) {
            .converged => std.debug.print("Converged after {d} steps (t = {d} s)\n", .{ step, t_end }),
            .diverged => std.debug.print("Diverged after {d} steps (t = {d} s)\n", .{ step, t_end }),
            .step_limit => std.debug.print("Hit step limit of {d} steps (t = {d} s)\n", .{ STEP_LIMIT, t_end }),
        }

        const end_tag = if (status == .converged) "steady" else @tagName(status);
        try write_field(self.io, try self.path(end_tag), field, .{ .method = self.method, .dt = self.dt, .t = t_end });
    }
};

// Explicit FTCS solver for one time step size dt
fn ftcs_solve(io: Io, dt: f64, nodes: []f64, nodes_swap: []f64) !SolveResult {
    // Fourier numbers, computed once before the loop
    const dx_coef: f64 = Thermal_diffusivity * dt / (dx * dx);
    const dy_coef: f64 = Thermal_diffusivity * dt / (dy * dy);

    // Boundaries live in both buffers so the swap never reads stale edges
    @memset(nodes, T_0);
    set_boundaries(nodes);
    @memcpy(nodes_swap, nodes);

    std.debug.print("=== FTCS dt = {d} s ===\n", .{dt});
    std.debug.print("Stability: d_x + d_y = {d:.3} (limit 0.5)\n", .{dx_coef + dy_coef});
    print_grid(nodes);

    var out: RunOutput = undefined;
    try out.open(io, "ftcs", dt, nodes);
    defer out.close();

    var current = nodes;
    var next = nodes_swap;
    var step: usize = 0;
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
        try out.logStep(step, current, tv);

        if (!std.math.isFinite(tv)) break .diverged;
        if (tv < CONSS) break .converged;
        if (step == STEP_LIMIT) break .step_limit;
    };

    try out.finish(status, step, current);
    return .{ .field = current, .steps = step, .status = status };
}

// sinh(a) / sinh(b) without overflow; needs 0 <= a <= b and b > 0.
// Uses sinh(x) = e^x (1 - e^(-2x)) / 2, so every exponent is <= 0.
fn sinh_ratio(a: f64, b: f64) f64 {
    return @exp(a - b) * (1.0 - @exp(-2.0 * a)) / (1.0 - @exp(-2.0 * b));
}

fn analytical_solution(alloc: std.mem.Allocator, length: f64, width: f64, t_bottom: f64, t_top: f64, dxloc: f64, dyloc: f64) ![]f64 {
    comptime if (T_2 != 0.0 or T_4 != 0) {
        @compileError("This is an invalid solution when T_2 or T_4 are non zero!");
    };
    // analytical is continuous but we want to compare to other methods so we evaluate at the same node locations
    const nodes: []f64 = alloc.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for analytical nodes matching numerical methods: {}\n", .{err});
        return err;
    }; // C
    set_boundaries(nodes);
    const pi: f64 = std.math.pi;
    const exit_tolerance: f64 = 1e-8; // very small change relative to the FTCS or ADI
    for (1..JMAX - 1) |j| {
        for (1..IMAX - 1) |i| {
            const k = idx(i, j);
            const x = dxloc * @as(f64, @floatFromInt(i)); // m
            const y = dyloc * @as(f64, @floatFromInt(j)); // m
            var T_a: f64 = 0.0; // C
            var T_b: f64 = 0.0; // C
            var tolerance = std.math.floatMax(f64);
            var m: f64 = 1.0;
            // Odd m only: 1 - cos(m pi) is 2 for odd m and 0 for even m (which would also
            // zero the bound below and stop the loop after one term), so 4 T / (m pi) is the amplitude
            while (tolerance > exit_tolerance) : (m += 2.0) {
                const b = m * pi * width / length; // shared sinh denominator argument
                const bound_a = 4.0 * t_bottom / (m * pi) * sinh_ratio(m * pi * (width - y) / length, b);
                const bound_b = 4.0 * t_top / (m * pi) * sinh_ratio(m * pi * y / length, b);
                const shape = std.math.sin(m * pi * x / length);
                T_a += bound_a * shape;
                T_b += bound_b * shape;
                // bound excludes the sine so a node where sin = 0 can't end the loop early
                tolerance = @max(@abs(bound_a), @abs(bound_b));
            }
            nodes[k] = T_a + T_b;
        }
    }
    return nodes;
}
// Solves Ax = d for tridiagonal A, in O(n) instead of O(n^3) for Gaussian elimination.
// x contains d on entry and is overwritten with the solution.
// Row i of A is: a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i].
// Only the first n = x.len entries of a, b, c and scratch are used, so the arrays may be
// longer; a[0] and c[n-1] are never read. scratch holds the modified superdiagonal.
// Stable without pivoting for diagonally dominant matrices.
// WARNING: does not handle periodic boundary conditions; see the Sherman-Morrison formula
// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
fn thomas(x: []f64, a: []const f64, b: []const f64, c: []const f64, scratch: []f64) !void {
    const n = x.len;
    if (n == 0) return error.EmptySystem;
    if (a.len < n or b.len < n or c.len < n or scratch.len < n) return error.LengthMismatch;

    // Forward sweep. Row 0 has no subdiagonal term, so it is handled before the loop.
    scratch[0] = c[0] / b[0];
    x[0] = x[0] / b[0];
    for (1..n) |i| {
        const denom = b[i] - a[i] * scratch[i - 1];
        if (i < n - 1) {
            scratch[i] = c[i] / denom; // B-8
        }
        x[i] = (x[i] - a[i] * x[i - 1]) / denom; // B-9
    }

    // Back substitution, counting down from n - 2. x[i] already holds the forward-sweep
    // value; the countdown is a while loop because an unsigned index can't go below 0.
    var i: usize = n - 1;
    while (i > 0) {
        i -= 1;
        x[i] -= scratch[i] * x[i + 1]; // B-5
    }
}

const testing = std.testing;

test "Thomas: dimensions" {
    var x: [2]f64 = .{ 8.0, 8.0 };
    const b: [1]f64 = .{4.0}; // shorter than x
    const c: [2]f64 = .{ 0.0, 0.0 };
    const a: [2]f64 = .{ 0.0, 0.0 };
    var scratch: [2]f64 = .{ 0.0, 0.0 };
    try testing.expectError(
        error.LengthMismatch,
        thomas(&x, &a, &b, &c, &scratch),
    );
}

test "Thomas: empty system" {
    var x: [0]f64 = .{};
    const a: [0]f64 = .{};
    const b: [0]f64 = .{};
    const c: [0]f64 = .{};
    var scratch: [0]f64 = .{};
    try testing.expectError(
        error.EmptySystem,
        thomas(&x, &a, &b, &c, &scratch),
    );
}

test "Thomas: trivial n of 1 case" {
    var x: [1]f64 = .{8.0};
    const b: [1]f64 = .{4.0};
    const c: [1]f64 = .{0.0};
    const a: [1]f64 = .{0.0};
    var scratch: [1]f64 = .{0.0};
    try thomas(&x, &a, &b, &c, &scratch);
    try testing.expectEqual(2.0, x[0]); // exact in floating point
}

test "Thomas: 2x2 system" {
    // [2 1; 1 2] x = [3; 3]  ->  x = [1; 1]
    var x: [2]f64 = .{ 3.0, 3.0 };
    const b: [2]f64 = .{ 2.0, 2.0 };
    const c: [2]f64 = .{ 1.0, 0.0 };
    const a: [2]f64 = .{ 0.0, 1.0 };
    var scratch: [2]f64 = undefined;
    try thomas(&x, &a, &b, &c, &scratch);
    try testing.expectApproxEqAbs(1.0, x[0], 1e-12);
    try testing.expectApproxEqAbs(1.0, x[1], 1e-12);
}

test "Thomas: 4x4 system" {
    var x: [4]f64 = .{ 1.0, -14, 26.5, 5 };
    const b: [4]f64 = .{ 5.0, 6.0, 7.0, 4.0 };
    const c: [4]f64 = .{ 2.0, -1.0, 3.0, 0.0 };
    const a: [4]f64 = .{ 0.0, 1, -2, 1 };
    var scratch: [4]f64 = undefined;
    try thomas(&x, &a, &b, &c, &scratch);
    const expected = [_]f64{ 1, -2, 3, 0.5 };
    for (0..x.len) |i| {
        try testing.expectApproxEqAbs(expected[i], x[i], 1e-12);
    }
}

test "Thomas: linear steady heat profile" {
    // -T[i-1] + 2 T[i] - T[i+1] = 0 inside, with T[0] = 40 and T[n-1] = 10.
    // The exact answer is linear: T[i] = 40 - 30 i / (n - 1).
    const n = 9;
    var x: [n]f64 = undefined;
    var a: [n]f64 = undefined;
    var b: [n]f64 = undefined;
    var c: [n]f64 = undefined;
    var scratch: [n]f64 = undefined;
    for (0..n) |i| {
        a[i] = -1.0;
        b[i] = 2.0;
        c[i] = -1.0;
        x[i] = 0.0;
    }
    // wall rows: identity, so x[0] = 40 and x[n-1] = 10
    a[0] = 0.0;
    b[0] = 1.0;
    c[0] = 0.0;
    x[0] = 40.0;
    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    x[n - 1] = 10.0;
    try thomas(&x, &a, &b, &c, &scratch);
    for (0..n) |i| {
        const exact = 40.0 - 30.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        try testing.expectApproxEqAbs(exact, x[i], 1e-10);
    }
}

test "Thomas: linear profile, walls folded into d" {
    // The setup the ADI sweeps use: interior unknowns only, wall values added into d.
    // 7 interior nodes between walls at 40 and 10, so T_i = 40 - 30 (i + 1) / 8.
    const n = 7;
    var x: [n]f64 = undefined;
    var a: [n]f64 = undefined;
    var b: [n]f64 = undefined;
    var c: [n]f64 = undefined;
    var scratch: [n]f64 = undefined;
    for (0..n) |i| {
        a[i] = -1.0;
        b[i] = 2.0;
        c[i] = -1.0;
        x[i] = 0.0;
    }
    a[0] = 0.0;
    c[n - 1] = 0.0;
    x[0] = 40.0; // -a * T_wall moved to the right-hand side
    x[n - 1] = 10.0;
    try thomas(&x, &a, &b, &c, &scratch);
    for (0..n) |i| {
        const exact = 40.0 - 30.0 * @as(f64, @floatFromInt(i + 1)) / 8.0;
        try testing.expectApproxEqAbs(exact, x[i], 1e-10);
    }
}

fn LineBuffers(comptime N: usize) type {
    return struct {
        const Self = @This();
        a: [N]f64 = undefined,
        b: [N]f64 = undefined,
        c: [N]f64 = undefined,
        d: [N]f64 = undefined,
        scratch: [N]f64 = undefined,
        fn zero(self: *Self) void {
            @memset(&self.a, 0.0);
            @memset(&self.b, 0.0);
            @memset(&self.c, 0.0);
            @memset(&self.d, 0.0);
            @memset(&self.scratch, 0.0);
        }
    };
}
// ADI solver for one time step size dt, using three field buffers:
//   current: the field at step n
//   star:    the half-step field T* after the x-sweep
//   next:    the field at step n + 1 after the y-sweep
// current and next are swapped each step, so they are local copies of the parameters
// (parameters are immutable in Zig). Read the final field from SolveResult.field.
// Uses the file-scope grid and boundary constants. MVP; parameterize if this gets refactored.
fn adi_solve(io: Io, dt: f64, nodes: []f64, star: []f64, nodes_next: []f64) !SolveResult {
    std.debug.assert(nodes.len == IMAX * JMAX and star.len == nodes.len and nodes_next.len == nodes.len);
    var current = nodes;
    var next = nodes_next;

    // set constants
    const nx = IMAX - 2;
    const ny = JMAX - 2;
    const d1: f64 = Thermal_diffusivity * 0.5 * dt / (dx * dx);
    const d2: f64 = Thermal_diffusivity * 0.5 * dt / (dy * dy);

    // init line buffers
    var x_lines: LineBuffers(nx) = .{};
    var y_lines: LineBuffers(ny) = .{};

    // Boundaries live in all three buffers so the sweeps and the swap never read stale edges
    @memset(current, T_0);
    set_boundaries(current);
    @memcpy(next, current);
    @memcpy(star, current);

    std.debug.print("=== ADI dt = {d} s ===\n", .{dt});
    std.debug.print("Half-step coefficients: d1 = {d:.3}, d2 = {d:.3} (unconditionally stable)\n", .{ d1, d2 });
    print_grid(current);

    // setup RunOutput for results
    var out: RunOutput = undefined;
    try out.open(io, "adi", dt, current);
    defer out.close();

    // init x_lines (k is the line index, not the grid column)
    for (0..nx) |k| {
        x_lines.a[k] = -d1;
        x_lines.c[k] = -d1;
        x_lines.b[k] = 1.0 + 2.0 * d1;
    }
    // a[0] and c[n-1] are never read by thomas; zeroed by convention
    x_lines.a[0] = 0.0;
    x_lines.c[nx - 1] = 0.0;
    // init y_lines
    for (0..ny) |k| {
        y_lines.a[k] = -d2;
        y_lines.c[k] = -d2;
        y_lines.b[k] = 1.0 + 2.0 * d2;
    }
    // a[0] and c[n-1] are never read by thomas; zeroed by convention
    y_lines.a[0] = 0.0;
    y_lines.c[ny - 1] = 0.0;
    var step: usize = 0;
    const status: Status = while (true) {
        var TV: f64 = 0.0;
        // x-sweep: implicit in x, explicit in y, one interior row at a time
        for (1..JMAX - 1) |j| {
            for (0..nx) |k| {
                const i = k + 1;
                x_lines.d[k] = d2 * current[idx(i, j - 1)] + (1 - 2 * d2) * current[idx(i, j)] + d2 * current[idx(i, j + 1)];
            }
            x_lines.d[0] += d1 * star[idx(0, j)]; // left wall
            x_lines.d[nx - 1] += d1 * star[idx(IMAX - 1, j)]; // right wall
            try thomas(&x_lines.d, &x_lines.a, &x_lines.b, &x_lines.c, &x_lines.scratch);
            for (0..nx) |k| {
                star[idx(k + 1, j)] = x_lines.d[k];
            }
        }
        // y-sweep: implicit in y, explicit in x, one interior column at a time
        for (1..IMAX - 1) |i| {
            for (0..ny) |k| {
                const j = k + 1;
                y_lines.d[k] = d1 * star[idx(i - 1, j)] + (1 - 2 * d1) * star[idx(i, j)] + d1 * star[idx(i + 1, j)];
            }
            y_lines.d[0] += d2 * next[idx(i, 0)]; // bottom wall
            y_lines.d[ny - 1] += d2 * next[idx(i, JMAX - 1)]; // top wall
            try thomas(&y_lines.d, &y_lines.a, &y_lines.b, &y_lines.c, &y_lines.scratch);
            for (0..ny) |k| {
                next[idx(i, k + 1)] = y_lines.d[k];
                TV += @abs(next[idx(i, k + 1)] - current[idx(i, k + 1)]);
            }
        }
        std.mem.swap([]f64, &current, &next);
        step += 1;
        try out.logStep(step, current, TV);

        if (!std.math.isFinite(TV)) break .diverged;
        if (TV < CONSS) break .converged;
        if (step == STEP_LIMIT) break .step_limit;
    };

    try out.finish(status, step, current);
    return .{ .field = current, .status = status, .steps = step };
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
    const time_step_sz = [_]f64{ 0.2, 1.0 }; //s
    const ftcs_nodes: []f64 = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for ftcs_nodes: {}\n", .{err});
        return err;
    }; // C
    const ftcs_nodes_swap: []f64 = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for ftcs_nodes_swap: {}\n", .{err});
        return err;
    }; // C

    for (time_step_sz) |dt| {
        const result = try ftcs_solve(io, dt, ftcs_nodes, ftcs_nodes_swap);
        print_grid(result.field);
    }
    const res_nodes_analytical = try analytical_solution(
        allocator,
        L,
        W,
        T_1,
        T_3,
        dx,
        dy,
    );
    std.debug.print("Analytical Steady State Solution\n", .{});
    print_grid(res_nodes_analytical);
    // try, not catch: a missing analytical.csv would silently break F6 and F7 in plot.py
    try write_field(
        io,
        OUT_DIR ++ "/analytical.csv",
        res_nodes_analytical,
        .{ .method = "analytical", .dt = 0, .t = 0 },
    );
    // ADI
    const adi_current = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for adi_current. Error: {}\n", .{err});
        return err;
    };
    const adi_star = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for adi_star. Error: {}\n", .{err});
        return err;
    };
    const adi_next = allocator.alloc(f64, IMAX * JMAX) catch |err| {
        std.debug.print("Error allocating memory for adi_next. Error: {}\n", .{err});
        return err;
    };
    for (time_step_sz) |dt| {
        const result: SolveResult = try adi_solve(io, dt, adi_current, adi_star, adi_next);
        print_grid(result.field);
    }
}

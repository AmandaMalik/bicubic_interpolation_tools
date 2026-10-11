import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { BicubicInterpolator, clampIndex, BoundaryMode } from "../src/index.js";

/**
 * Bicubic interpolation of a constant field must reproduce the constant
 * everywhere, including across boundaries, regardless of mode. This is the
 * most important correctness invariant: the kernel is a partition of unity.
 */
function expectConstant(mode) {
  const nx = 5;
  const ny = 4;
  const value = 7.5;
  const data = new Array(nx * ny).fill(value);
  const interp = new BicubicInterpolator(data, nx, ny, mode);
  const points = [
    [0, 0],
    [nx - 1, ny - 1],
    [2.5, 1.5],
    [-0.3, 1.2],
    [nx - 0.5, ny - 0.5],
    [2.5, -0.5],
  ];
  for (const [x, y] of points) {
    const got = interp.sample(x, y);
    assert.ok(
      Math.abs(got - value) < 1e-9,
      `mode ${mode} at (${x},${y}) got ${got}, expected ${value}`,
    );
  }
}

describe("clampIndex", () => {
  it("reflects about edges excluding the edge sample", () => {
    assert.equal(clampIndex(-1, 5, BoundaryMode.REFLECT), 1);
    assert.equal(clampIndex(-2, 5, BoundaryMode.REFLECT), 2);
    assert.equal(clampIndex(5, 5, BoundaryMode.REFLECT), 3);
    assert.equal(clampIndex(6, 5, BoundaryMode.REFLECT), 2);
    assert.equal(clampIndex(7, 5, BoundaryMode.REFLECT), 1);
    assert.equal(clampIndex(0, 5, BoundaryMode.REFLECT), 0);
    assert.equal(clampIndex(4, 5, BoundaryMode.REFLECT), 4);
  });

  it("mirrors including the edge sample", () => {
    assert.equal(clampIndex(-1, 5, BoundaryMode.MIRROR), 0);
    assert.equal(clampIndex(-2, 5, BoundaryMode.MIRROR), 1);
    assert.equal(clampIndex(5, 5, BoundaryMode.MIRROR), 4);
    assert.equal(clampIndex(6, 5, BoundaryMode.MIRROR), 3);
  });

  it("clamps to the nearest in-range index", () => {
    assert.equal(clampIndex(-3, 5, BoundaryMode.CLAMP), 0);
    assert.equal(clampIndex(10, 5, BoundaryMode.CLAMP), 4);
    assert.equal(clampIndex(2, 5, BoundaryMode.CLAMP), 2);
  });

  it("returns -1 for out-of-range in zero mode and the index otherwise", () => {
    assert.equal(clampIndex(-1, 5, BoundaryMode.ZERO), -1);
    assert.equal(clampIndex(5, 5, BoundaryMode.ZERO), -1);
    assert.equal(clampIndex(2, 5, BoundaryMode.ZERO), 2);
  });

  it("handles n === 1 by returning 0 for reflect/mirror/clamp", () => {
    assert.equal(clampIndex(-4, 1, BoundaryMode.REFLECT), 0);
    assert.equal(clampIndex(5, 1, BoundaryMode.MIRROR), 0);
    assert.equal(clampIndex(-2, 1, BoundaryMode.CLAMP), 0);
    assert.equal(clampIndex(3, 1, BoundaryMode.ZERO), -1);
  });

  it("rejects unknown modes and bad dimensions", () => {
    assert.throws(() => clampIndex(0, 5, "nope"), TypeError);
    assert.throws(() => clampIndex(0, -1, BoundaryMode.CLAMP), RangeError);
    assert.throws(() => clampIndex(0, 1.5, BoundaryMode.CLAMP), RangeError);
  });
});

describe("BicubicInterpolator", () => {
  it("reproduces a constant field in every mode", () => {
    expectConstant(BoundaryMode.REFLECT);
    expectConstant(BoundaryMode.MIRROR);
    expectConstant(BoundaryMode.CLAMP);
    expectConstant(BoundaryMode.ZERO);
  });

  it("reproduces a linear ramp in x exactly at interior points", () => {
    const nx = 6;
    const ny = 6;
    const data = new Array(nx * ny);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        data[y * nx + x] = 2 * x + 3;
      }
    }
    const interp = new BicubicInterpolator(data, nx, ny, BoundaryMode.REFLECT);
    // Catmull-Rom reproduces cubics, hence also linears, at interior points.
    for (const x of [1.25, 2.5, 3.75]) {
      for (const y of [1, 2, 3]) {
        const got = interp.sample(x, y);
        const want = 2 * x + 3;
        assert.ok(Math.abs(got - want) < 1e-9, `(${x},${y}) got ${got} want ${want}`);
      }
    }
  });

  it("returns the exact node value when sampling on a grid point", () => {
    const nx = 4;
    const ny = 3;
    const data = [];
    for (let i = 0; i < nx * ny; i++) data.push(i * 1.5);
    const interp = new BicubicInterpolator(data, nx, ny, BoundaryMode.CLAMP);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const got = interp.sample(x, y);
        const want = data[y * nx + x];
        assert.ok(Math.abs(got - want) < 1e-9, `node (${x},${y}) got ${got} want ${want}`);
      }
    }
  });

  it("produces the same result for x then y as y then x symmetry on a symmetric field", () => {
    const n = 5;
    const data = new Array(n * n);
    // Build a field symmetric under x <-> y: data[i,j] = f(i)+f(j) is symmetric.
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        data[y * n + x] = (x - 2) * (x - 2) + (y - 2) * (y - 2);
      }
    }
    const interp = new BicubicInterpolator(data, n, n, BoundaryMode.REFLECT);
    const a = interp.sample(1.5, 2.5);
    const b = interp.sample(2.5, 1.5);
    assert.ok(Math.abs(a - b) < 1e-9, `symmetric field gave ${a} vs ${b}`);
  });

  it("zero mode returns 0 far outside the grid", () => {
    const data = [1, 2, 3, 4];
    const interp = new BicubicInterpolator(data, 2, 2, BoundaryMode.ZERO);
    // At (10,10) every tap falls outside, so the normalised sum is 0 by definition.
    assert.equal(interp.sample(10, 10), 0);
  });

  it("clamp mode extends the edge value", () => {
    const data = [1, 2, 3, 4];
    const interp = new BicubicInterpolator(data, 2, 2, BoundaryMode.CLAMP);
    // With CLAMP and a constant extrapolation request, the edge value reproduces.
    assert.ok(Math.abs(interp.sample(-5, 0) - 1) < 1e-9);
    assert.ok(Math.abs(interp.sample(10, 1) - 4) < 1e-9);
  });

  it("rejects non-finite coordinates", () => {
    const interp = new BicubicInterpolator([1, 2, 3, 4], 2, 2);
    assert.throws(() => interp.sample(NaN, 0), RangeError);
    assert.throws(() => interp.sample(0, Infinity), RangeError);
  });

  it("rejects mismatched data length", () => {
    assert.throws(() => new BicubicInterpolator([1, 2, 3], 2, 2), RangeError);
    assert.throws(() => new BicubicInterpolator([], 0, 0), RangeError);
  });

  it("defaults to reflect boundary mode", () => {
    const interp = new BicubicInterpolator([1, 2, 3, 4], 2, 2);
    assert.equal(interp._mode, BoundaryMode.REFLECT);
  });

  it("rejects unknown boundary mode", () => {
    assert.throws(() => new BicubicInterpolator([1, 2, 3, 4], 2, 2, "wrap"), TypeError);
  });
});

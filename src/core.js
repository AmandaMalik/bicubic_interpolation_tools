/**
 * BoundaryMode encodes how coordinates outside [0, n-1] are resolved when
 * sampling the input grid during convolution. We expose the four options the
 * kernel actually needs rather than overloading a single "edge" flag: each
 * mode produces measurably different output near the borders, so conflating
 * them would make the behaviour impossible to reason about.
 */
export const BoundaryMode = Object.freeze({
  /**
   * Mirror samples about the nearest edge index, excluding the edge itself:
   * -1 -> 1, -2 -> 2, n -> n-2. Produces a continuous derivative across the
   * boundary because the edge value is not duplicated, which is what we want
   * for smooth image-like data.
   */
  REFLECT: "reflect",
  /**
   * Mirror samples including the edge value: -1 -> 0, -2 -> 1, n -> n-1.
   * Simpler than REFLECT but creates a zero derivative at the boundary.
   */
  MIRROR: "mirror",
  /**
   * Clamps out-of-range indices to the nearest in-range index, duplicating
   * edge samples. Useful when the edges are genuinely constant (e.g. padding
   * a tile) but produces a discontinuous derivative at the boundary.
   */
  CLAMP: "clamp",
  /**
   * Treats out-of-range samples as zero. Appropriate when the grid is embedded
   * in a zero-valued field and the boundary is physically meaningful.
   */
  ZERO: "zero",
});

/**
 * Resolve a (possibly fractional) index into a source-grid integer index,
 * applying the requested boundary handling.
 *
 * Written separately from BicubicInterpolator so the boundary logic is
 * testable in isolation; the convolution kernel below depends on this being
 * correct for all four modes.
 */
export function clampIndex(i, n, mode) {
  if (!Number.isInteger(n) || n < 0) {
    throw new RangeError(`grid dimension must be a non-negative integer, got ${String(n)}`);
  }
  if (n === 0) {
    // An empty axis has no valid index; the caller cannot use this result.
    return -1;
  }
  if (mode === BoundaryMode.ZERO) {
    if (i < 0 || i >= n) {
      return -1;
    }
    return i;
  }
  if (mode === BoundaryMode.CLAMP) {
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
  }
  if (mode === BoundaryMode.REFLECT) {
    // Period 2*(n-1) when n > 1; for n === 1 every index maps to 0.
    if (n === 1) return 0;
    const period = 2 * (n - 1);
    let r = ((i % period) + period) % period; // normalise to [0, period)
    if (r >= n) {
      r = period - r;
    }
    return r;
  }
  if (mode === BoundaryMode.MIRROR) {
    // Period 2*n; mirror including the edge sample.
    const period = 2 * n;
    let r = ((i % period) + period) % period;
    if (r >= n) {
      r = period - 1 - r;
    }
    return r;
  }
  throw new TypeError(`unknown boundary mode: ${String(mode)}`);
}

/**
 * Catmull-Rom (Keys) cubic convolution weight for a single sample at signed
 * distance d from the interpolation point, where d is in {-2,-1,0,1,2} for the
 * nearest-grid case but the formula is general. We use the standard piecewise
 * cubic because it is C^1 continuous and reproduces a cubic exactly — the
 * usual choice for image resampling and the one a reader will expect.
 */
function cubicWeight(t) {
  const at = Math.abs(t);
  if (at < 1) {
    return 1.5 * at * at * at - 2.5 * at * at + 1;
  }
  if (at < 2) {
    return -0.5 * at * at * at + 2.5 * at * at - 4 * at + 2;
  }
  return 0;
}

function assertGrid(data, nx, ny) {
  if (!Array.isArray(data)) {
    throw new TypeError("data must be a dense Array");
  }
  if (!Number.isInteger(nx) || !Number.isInteger(ny)) {
    throw new TypeError("nx and ny must be integers");
  }
  if (nx < 0 || ny < 0) {
    throw new RangeError("nx and ny must be non-negative");
  }
  const expected = nx * ny;
  if (expected !== data.length) {
    throw new RangeError(`data length ${data.length} does not match nx*ny=${expected}`);
  }
}

export class BicubicInterpolator {
  /**
   * @param {number[]} data row-major grid, length nx*ny, indexed as
   *   data[y * nx + x] (y is the row, x the column). Row-major is chosen
   *   because it matches typed-array conventions and makes the inner loops
   *   access memory sequentially in x.
   * @param {number} nx  number of columns
   * @param {number} ny  number of rows
   * @param {string} mode boundary mode, one of BoundaryMode
   */
  constructor(data, nx, ny, mode = BoundaryMode.REFLECT) {
    assertGrid(data, nx, ny);
    this._data = data;
    this._nx = nx;
    this._ny = ny;
    this._mode = mode;
    if (nx === 0 || ny === 0) {
      throw new RangeError("grid must contain at least one cell");
    }
    if (!Object.values(BoundaryMode).includes(mode)) {
      throw new TypeError(`unknown boundary mode: ${String(mode)}`);
    }
  }

  /**
   * Sample the grid at fractional coordinates (x, y). Out-of-range input
   * coordinates are forwarded to the boundary handler, so the result is
   * defined everywhere — this is deliberate; callers interpolating into a
   * larger field should not have to special-case the border.
   */
  sample(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new RangeError(`coordinates must be finite, got (${x}, ${y})`);
    }
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const dx = x - x0;
    const dy = y - y0;

    let total = 0;
    let totalW = 0;
    for (let j = -1; j <= 2; j++) {
      const gy = clampIndex(y0 + j, this._ny, this._mode);
      if (gy < 0) continue; // ZERO mode outside grid
      const wy = cubicWeight(j - dy);
      for (let i = -1; i <= 2; i++) {
        const gx = clampIndex(x0 + i, this._nx, this._mode);
        if (gx < 0) continue;
        const w = wy * cubicWeight(i - dx);
        total += w * this._data[gy * this._nx + gx];
        totalW += w;
      }
    }
    // In ZERO mode with the interpolation point itself outside the grid and
    // far enough that all four taps fall outside, totalW is 0. We return 0
    // rather than dividing by zero — the field is zero there by definition.
    if (totalW === 0) return 0;
    // Normalising by totalW keeps the partition-of-unity property when some
    // taps are zeroed at the boundary; without this the ZERO mode would
    // darken near edges, which is rarely what a caller wants.
    return total / totalW;
  }
}

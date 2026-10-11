# bicubic_interpolation_tools

Interpolates values on a two-dimensional regular grid using bicubic convolution (Catmull-Rom / Keys kernel) with four selectable boundary-handling modes.

## Usage

```js
import { BicubicInterpolator, BoundaryMode } from "./src/index.js";

// Row-major grid: data[y * nx + x], here a 4x3 ramp.
const nx = 4;
const ny = 3;
const data = [];
for (let y = 0; y < ny; y++)
  for (let x = 0; x < nx; x++)
    data.push(x + y * nx);

const interp = new BicubicInterpolator(data, nx, ny, BoundaryMode.REFLECT);
console.log(interp.sample(1.5, 1.5)); // -> 8.5
```

Exports: `BicubicInterpolator` (class), `BoundaryMode` (frozen enum), `clampIndex` (the boundary resolver, exported for testing).

## Why

Bicubic convolution gives a C¹-continuous surface from a discrete grid, which matters when the interpolated values feed a downstream process sensitive to derivative discontinuities (normals, gradients, optimisation). The cost is some overshoot near sharp transitions — a known trade-off versus bilinear interpolation, which is C⁰ only but never overshoots.

## Boundary modes

`REFLECT` (default), `MIRROR`, `CLAMP`, `ZERO`. `REFLECT` mirrors about the edge *without* duplicating the edge sample, so the derivative stays continuous across the boundary; `CLAMP` duplicates the edge and is the right pick when the edges are genuinely constant padding; `ZERO` treats out-of-range taps as zero and normalises by the remaining weight, so it does not darken near the border.

## Edge you will hit

Sampling exactly on a grid node returns the node value to within floating-point error. With `ZERO` mode, points whose entire 4×4 support falls outside the grid return `0` by definition rather than `NaN` — if you need to distinguish "outside" from "zero value", check the coordinates yourself before calling `sample`.

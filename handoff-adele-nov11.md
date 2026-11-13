# Handoff Documentation: Curb Height Estimation & 3D Mesh Reconstruction

Project: KITTI-360 | Reference: `heights_per_slice.ipynb`

## 1\. Project overview

The current pipeline estimates curb heights at multiple locations within a KITTI-360 frame using LiDAR point clouds and semantic segmentation.

The next goal is to combine these individual fitted cross-sections into a continuous 3D surface, allowing us to interpolate curb heights at locations where we don't have direct estimates.

Current pipeline → Proposed extension

KITTI-360 point cloud

World XYZ coordinates + semantic labels

Depth bands and boundary candidates

Local road/sidewalk regions

Fit a road–curb–sidewalk height profile

Quadratic road + linear ramp + linear sidewalk

Interpolate between fitted profiles

Proposed next development step

Continuous 3D mesh

Estimate curb geometry and height at new positions

## 2\. How the current implementation works

The notebook operates on one KITTI-360 frame and its associated static point cloud.

1. Load and project the data. Load the point cloud, camera calibration, pose, and image. Project world points into the image and filter approximately for visible points.
2. Split points into camera-depth bands. Select target depths between the 10th and 90th percentiles of visible ground-point depths. Each band covers ±2 metres in camera depth.
3. Find boundary candidates. For each band, generate up to `candidate_num` road/sidewalk boundary anchors and use the first candidate whose estimator succeeds.
4. Estimate the local curb geometry. The boundary estimator uses PCA to construct a local coordinate system. It bins points across the curb, calculates median elevations, and fits a piecewise height function.
5. Store fitted results. Each successful estimate contains the estimated height, transition information, local geometry, and fitted curve.

### Mathematical model

Each cross-section uses a coordinate $s$ measuring horizontal distance from road toward sidewalk. The fit center is $c$, and $d=s-c$.

The fitted road and sidewalk are:

$$
z_{\text{road}}(s)=b+m_r d+qd^2
$$

$$
z_{\text{sidewalk}}(s)=b+h+m_s d
$$

For a nonzero ramp width $w$, the transition connects these models linearly between $c-w/2$ and $c+w/2$.

The fitted coefficients are:

$$
\beta=[b,m_r,h,m_s,q]
$$

This gives us a compact, continuous height function across each local road/sidewalk boundary rather than only one curb-height measurement.

## 3\. Current data available for reconstruction

The notebook's `run_experiment()` returns `results_by_distance`, a list of dictionaries. Each entry contains:

| Field | Meaning |
| --- | --- |
| `depth` | Target camera depth in metres |
| `height_cm` | Estimated step height in centimetres |
| `ramp_slope` | Currently populated using `sidewalk_slope` |
| `fit` | Fitted parameter dictionary, if returned by estimator |
| `result` | Complete local boundary-estimator output |

The boundary estimator's result includes local information such as:

| Field | Meaning |
| --- | --- |
| `center` | XY anchor position |
| `mean` | PCA local origin in XY |
| `normal` | Unit vector pointing road → sidewalk |
| `pts`, `z`, `lab` | Selected cross-section positions, elevations, and labels |
| `bins` | Binned position/elevation profile |
| `s_step` | Fitted transition center |
| `step_cm` | Fitted vertical offset in centimetres |
| `ramp_width_cm` | Fitted transition width |
| `fit_curve` | Sampled fitted height curve |

Important implementation detail: The notebook reads `result.get("fit")`, but the estimator return statement you shared earlier did not contain a `fit` field. Ensure the estimator actually returns `fit=fit`, otherwise the reconstruction code will not have the fitted coefficients.

Similarly, `ramp_slope` in the experiment is currently the sidewalk slope, not the slope of the curb ramp. I recommend renaming it `sidewalk_slope` to avoid confusion.

## 4\. Recommended approach to building the mesh

I'd start with sampling the existing fitted curves in world coordinates, then interpolating between neighboring slices.

This lets us reuse the current model without immediately developing a more complicated global optimizer.

Conceptual illustration: fitted cross-sections can act as constraints for a surface spanning the road, curb, and sidewalk.

### Step A — Reconstruct each fitted cross-section

Each fit already defines a function:

$$
z_i(s)=\text{design\_matrix}(s,c_i,w_i)\beta_i
$$

Sample this function at regularly spaced positions along the cross-section.

For example:

python

```python
s = np.linspace(-2, 2, 100)
z = design_matrix(s, fit["c"], fit["width"]) @ fit["beta"]
```

This gives 100 points describing the modeled profile.

### Step B — Convert each cross-section into world XYZ

Each cross-section has its own origin and normal vector.

The original global XY positions can be recovered using:

$$
\begin{bmatrix}X\\Y\end{bmatrix}
=\mu_i+s\mathbf{n}_i
$$

where $\mu_i$ is `result["mean"]` and $\mathbf n_i$ is `result["normal"]`.

python

```python
xy = result["mean"] + s[:, None] * result["normal"]

points_3d = np.column_stack([
    xy,
    z
])
```

This gives a set of actual world-coordinate points along the fitted curve.

This transformation is particularly important: you should not treat camera depth as the world position of a slice. Camera depth is used for selecting bands, whereas the mesh geometry should be built from world coordinates.

### Step C — Interpolate between neighboring cross-sections

For a first prototype, I'd suggest using the simplest approach:

- Sort neighboring cross-sections spatially along the curb.
- Sample each cross-section at comparable distances from its fitted curb center.
- Interpolate the resulting heights between neighboring profiles.
- Join adjacent sampled points into triangles.

For two cross-sections, interpolation at a relative position $\lambda$ can use:

$$
z(s,\lambda)=(1-\lambda)z_1(s)+\lambda z_2(s)
$$

where $0\le\lambda\le1$.

However, because each cross-section may have a slightly different orientation, the sampled profiles should first be aligned by their actual curb positions and local normals rather than assuming their `s=0` locations correspond.

For an initial prototype, you can use linear interpolation. Later, cubic splines or radial basis functions could provide smoother surfaces.

### Step D — Build the triangular mesh

Once the interpolated grid is constructed, use adjacent points as mesh vertices.

For a rectangular grid, each four-vertex cell becomes two triangles.

Libraries worth looking at:

- `scipy.interpolate` — interpolation between sampled profiles
- `scipy.spatial` — triangulation and spatial nearest-neighbor queries
- `open3d` — mesh creation and visualization
- `pyvista` — convenient 3D surface visualization

## 5\. Minimal code to get started

This is a useful first function for converting an existing fit into world-coordinate samples:

python

```python
def sample_fit(result, n_samples=100, margin=1.0):
    fit = result["fit"]

    c = fit["c"]
    width = fit["width"]

    # Sample the region around the fitted curb
    s = np.linspace(
        c - width / 2 - margin,
        c + width / 2 + margin,
        n_samples
    )

    # Evaluate fitted elevation
    A = analyze_boundary.design_matrix(s, c, width)
    z = A @ fit["beta"]

    # Convert local coordinates to world XYZ
    xy = result["mean"] + s[:, None] * result["normal"]

    return np.column_stack((xy, z))
```

This assumes that `design_matrix` is accessible from `analyze_boundary` and that the estimator returns its fitted coefficients.

To test:

python

```python
profiles = [
    sample_fit(item["result"])
    for item in results
    if item["result"] is not None
]
```

Now `profiles` is a list of arrays, each containing sampled XYZ coordinates.

You could visualize these first before implementing interpolation.

## 6\. Things to watch out for

These are the main issues I would highlight to whoever takes over the project.

| Issue | Why it matters |
| --- | --- |
| Different PCA directions | Each cross-section has its own normal, so profiles need spatial alignment |
| Missing or failed fits | Some depth bands may not contain a usable boundary |
| Overlapping depth bands | Nearby fits may contain many of the same LiDAR points |
| Curved boundaries | A simple straight-line connection may not follow the curb |
| Road elevation changes | Interpolation should preserve the underlying road slope |
| Flat regions | Not every road/sidewalk transition contains a curb |
| Semantic discontinuities | Avoid joining unrelated sidewalks or different curb segments |

I'd especially avoid interpolating only the estimated curb heights. You have much more useful information available: the full quadratic road, ramp, and sidewalk models.

Interpolating the actual profiles should give a more informative surface.

Also, the current experiment uses the first successful boundary candidate per depth band, not necessarily the best one. Before meshing, it may be worth rejecting poor fits or spatially inconsistent candidates to avoid reconstructing the wrong boundary.

## 7\. Suggested development milestones

Milestone 1 — Visualize fitted profiles in 3D

Extract fit coefficients and transform all existing cross-sections into world XYZ. Display them alongside the original LiDAR cloud to verify positioning.

Milestone 2 — Construct a basic mesh

Order neighboring fitted profiles, align them to the fitted curb centers, interpolate between them, and connect samples into a triangle mesh.

Milestone 3 — Query new curb-height estimates

At arbitrary positions along the reconstructed curb, estimate road and sidewalk elevation differences from the mesh.

Milestone 4 — Evaluate interpolation accuracy

Withhold selected fitted slices, reconstruct the surface using the others, and compare predicted curb heights against the withheld estimates. Where available, compare against independent ground-truth measurements as well.

## 8\. Recommended starting point

I would prioritize these three tasks:

1. Make sure each `boundary_estimator()` result stores `fit`, `mean`, `normal`, and `s_step`. These are essential for reconstructing the fitted profiles in world coordinates.
2. Implement `sample_fit()` and plot all fitted cross-sections in 3D, overlaid on the original LiDAR points. Confirm the profiles are aligned before attempting interpolation.
3. Build the simplest possible mesh between two neighboring profiles first. Once that works, generalize to all profiles and more complex boundaries.

The main technical challenge is not necessarily the interpolation algorithm; it is ensuring that the individual cross-sections correspond to the same physical curb and that their coordinates are aligned properly.

If that alignment is reliable, the first version of the mesh can be relatively simple.

If you want, I can:

- Provide detailed documentation and next steps for 3D mesh creation from fit models
- Suggest code additions to transform fit outputs for mesh interpolation
- Recommend approach and important considerations for mesh interpolation of curb heights
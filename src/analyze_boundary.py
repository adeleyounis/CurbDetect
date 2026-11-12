# imports
import numpy as np
from scipy.spatial import cKDTree

def design_matrix(x, c, width):
    """
        Road: b + mr * dx + q * dx**2 (quadratic due to water management)
        Sidewalk: b + h + ms * dx (flat and linear)
        b: Road height at the curb center
        mr: road slope 
        h : vertical offset between road and sidewalk at the center
        ms: sidewalk slope
        q: road curvature
    """
    x = np.asarray(x, dtype=float)
    dx = x - c
    one, zero = np.ones_like(x), np.zeros_like(x)

    road_features = np.column_stack((one, dx, zero, zero, dx ** 2))
    side_features = np.column_stack((one, zero, one, dx, zero))

    if width == 0:  # quadratic road -> step -> linear sidewalk
        return np.where((x >= c)[:, None], side_features, road_features)

    # no real instantaneous curb, always a slight lsope 
    lo, hi = c - width / 2, c + width / 2
    alpha = np.clip((x - lo) / width, 0.0, 1.0)

    # evaluate the road at the ramp's start and sidewalk at its end
    road_at_lo = np.array([1.0, -width / 2, 0.0, 0.0, width ** 2 / 4])
    side_at_hi = np.array([1.0, 0.0, 1.0, width / 2, 0.0])

    # linearly interpolate the exndpoints so the ramp stays linear
    ramp_features = ((1 - alpha[:, None]) * road_at_lo + alpha[:, None] * side_at_hi)

    A = np.where(
        (x < lo)[:, None], road_features, 
        np.where((x > hi)[:, None], side_features, 
        ramp_features) # between low to high - transition region
        )
    return A

def try_fit(width_values, centers, meds, cnt):
        best = None
        sqrt_w = np.sqrt(cnt / np.mean(cnt))

        for i in range(3, len(centers) - 3): # avoid transition regions tooo close to bin edges
            # midpoint of bins -> corresponds to the centre of the transtion area
            c = 0.5 * (centers[i - 1] + centers[i])

            for width in width_values:
                A = design_matrix(centers, c, width)
                beta, _, _, _ = np.linalg.lstsq(A * sqrt_w[:, None], meds * sqrt_w)

                predicted = A @ beta
                residual = meds - predicted
                weighted_mse = np.sum(cnt * residual ** 2) / np.sum(cnt)

                if best is None or weighted_mse < best["mse"]:
                    best = dict(c=c, width=width, beta=beta, mse=weighted_mse)

        return best

def pca(edge, road, side, points_xyz):
    mu = edge.mean(0)
    w, v = np.linalg.eigh(np.cov((edge - mu).T))
    t = v[:, np.argmax(w)]
    n = np.array([-t[1], t[0]])
    if (side.mean(0) - road.mean(0)) @ n < 0: # want normal from road to sidewalk
        n = -n

    # restrict to a strip around the center
    along = (points_xyz - mu) @ t # distance parallel to curb
    along = np.abs(along) < 10 # limit parallel points to be wtithin a certain strip
    across = (points_xyz - mu) @ n # distance perp to curb
    return across, along, mu, n

def assess_fit(fit, centers):
    curve_x = np.linspace(centers.min(), centers.max(), 400) # distance in cm
    curve_z = design_matrix(curve_x, fit['c'], fit['width']) @ fit['beta'] # world z in cm
    return (curve_x, curve_z)

def boundary_candidates(cloud, road_id, sidewalk_id, n=30, radius=0.3, min_spacing=5.0, max_road_pts=200_000, seed=None):
    # seed=None keeps the old (random every run) behaviour; pass an int for reproducible runs
    random_gen = np.random.default_rng(seed)

    xyz = cloud["xyz"][:, :2]
    semantic = cloud["semantic"]
    road = xyz[semantic == road_id] # n x m array
    sidewalk = xyz[semantic == sidewalk_id] # n x m array

    if len(road) == 0 or len(sidewalk) == 0:
        return np.empty((0, 2))
    if len(road) > max_road_pts:
        road = road[random_gen.choice(len(road), max_road_pts, replace=False)] # downsample

    distance, _ = cKDTree(sidewalk).query(road, distance_upper_bound=radius) # spatial search mapping closest points of sidewalk to query - within a radius
    edge = road[np.isfinite(distance)]
    random_gen.shuffle(edge) # to not prioritize ordered points

    picked = []
    for p in edge: 
        if all(np.hypot(*(p - q)) > min_spacing for q in picked): # euclidian distance > min spacing threshold between pts
            picked.append(p)
            if len(picked) == n: # only want n closest points
                break
    return np.array(picked)

def boundary_estimator(cloud, center_xy, road_id, sidewalk_id, window_width=6.0, edge_radius=0.5, bin_w=0.05):
    xyz = cloud["xyz"]
    semantic = cloud["semantic"]
    center_xy = np.asarray(center_xy, float) # boundary should anchor around a candidate 

    # window around the centered point
    window = (np.abs(xyz[:, 0] - center_xy[0]) < window_width/2) & (np.abs(xyz[:, 1] - center_xy[1]) < window_width/2)
    window &= np.isin(semantic, [road_id, sidewalk_id])
    idx = np.where(window)[0]

    point_coords = xyz[idx, :2]
    z = xyz[idx, 2]

    road = point_coords[semantic[idx] == road_id]
    side = point_coords[semantic[idx] == sidewalk_id]
    if len(road) == 0 or len(side) == 0:  # guard: nothing to fit
        return None

    # label-edge points - road points that are within a radius of the sidewalk points
    dist, _ = cKDTree(side).query(road, distance_upper_bound=edge_radius)
    edge = road[np.isfinite(dist)]
    if len(edge) < 3:  # guard: PCA needs a few edge points
        return None

    # PCA - want the largest variance to be our boundary
    across, along, mean, normal = pca(edge, road, side, point_coords)
    boundary_pt_pos, height, label = across[along], z[along], semantic[idx][along]
    
    road_pos = boundary_pt_pos[label == road_id]
    sidewalk_pos = boundary_pt_pos[label == sidewalk_id]

    if len(road_pos) == 0 or len(sidewalk_pos) == 0: 
        return None

    # identify edges
    edges = np.arange(boundary_pt_pos.min(), boundary_pt_pos.max() + bin_w, bin_w)
    bins = np.digitize(boundary_pt_pos, edges)

    centers, meds, cnt = [], [], []
    for k in np.unique(bins):
        m = bins == k # boolean mask the bins
        if m.sum() >= 3:
            centers.append(boundary_pt_pos[m].mean())
            meds.append(np.median(height[m]))
            cnt.append(m.sum())
    centers, meds, cnt = map(np.array, (centers, meds, cnt))

    # widths are in meters, from 2 bins to at most 1 meter.
    max_width = min(1.0, (centers.max() - centers.min()) / 2)
    widths = np.arange(2 * bin_w, max_width + 1e-9, 2 * bin_w)

    fit = try_fit(widths, centers, meds, cnt)
    if fit is None:  # guard: fewer than 7 usable bins, try_fit never ran
        return None
    step_cm = fit['beta'][2] * 100  # signed offset at the transition midpoint

    return dict(center=center_xy, mean=mean, normal=normal, pts=boundary_pt_pos,
                z=height, lab=label, bins=(centers, meds),
                s_step=fit['c'], step_cm=step_cm, sidewalk_slope=fit['beta'][3],
                ramp_width_cm=fit['width'] * 100,
                fit_curve=assess_fit(fit, centers),
                fit=fit,                # NEW: c, width, beta, mse -> needed to rebuild the profile
                point_idx=idx[along])   # NEW: indices (into the cloud passed in) of pts/z/lab, same order

def get_curb_side(res, cam_to_world, proj_matrix):
    z = np.median(res["z"])

    points = np.array([
        [res["mean"][0], res["mean"][1], z],
        [res["mean"][0] + 0.25 * res["normal"][0],
         res["mean"][1] + 0.25 * res["normal"][1], z]
    ])

    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=float))
    P = np.asarray(proj_matrix, dtype=float)

    pc = (points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3])
    uvw = pc @ P[:, :3].T + P[:, 3]

    if np.any(uvw[:, 2] <= 0):
        return None

    u = uvw[:, 0] / uvw[:, 2]

    return "left" if u[1] < u[0] else "right"
import numpy as np
from plyfile import PlyData

def get_img_path(root, seq, cam_id, frame):
    return root / "data_2d_raw" / seq / f"image_0{cam_id}" / "data_rect" / f"{frame:010d}.png"  

def get_chunks_for_frame(chunks, frame):
    return [c for c in chunks if c[0] <= frame <= c[1]]

def load_perspective_calibration(path, cam_id=0):
    """load the cameras perspective intrinsics 
    stored in /KITTI-360/calibration/perspective.txt"""
    
    dict = {}
    with open(path / "calibration/perspective.txt") as file:
        for line in file:
            k, v = line.split(":", 1)
            dict[k.strip()] = np.array(v.split())

    proj_matrix = dict[f"P_rect_0{cam_id}"].reshape(3,4)
    width = dict[f"S_rect_0{cam_id}"][0]
    height = dict[f"S_rect_0{cam_id}"][1]

    return proj_matrix, int(float(width)), int(float(height))

def load_cam0_to_world(path, seq):
    """load in the camera sessions pose mapping 
    to the world coordinates"""

    file_path = path / "data_poses" / seq / "cam0_to_world.txt"
    data = np.loadtxt(file_path) 
    dict = {
        int(r[0]) : r[1:].reshape(4,4) for r in data
    }
    return dict

def load_ply(path):
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    out = {"xyz": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)}
    if "red" in names:
        out["rgb"] = np.stack([v["red"], v["green"], v["blue"]], 1)
    for k in ("semantic", "instance", "confidence", "visible"):
        if k in names:
            out[k] = np.asarray(v[k])
    print(f"{path.name}: {len(out['xyz']):,} points, fields: {names}")
    return out

def load_cloud_for_frame(chunks, frame):
    parts = [load_ply(c[2]) for c in get_chunks_for_frame(chunks, frame)]
    keys = set.intersection(*[set(p) for p in parts])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}

def list_static_chunks(path, split, seq):
    file_path = path / "data_3d_semantics" / split / seq / "static" 
    files = sorted(file_path.glob("*.ply"))
    chunks = []
    for file in files:
        start, end = file.stem.split("_")
        chunks.append((int(start), int(end), file))
    return chunks

def project(xyz_world, cam_to_world, proj_matrix, W, H, min_depth=0.5):
    world_to_cam = np.linalg.inv(cam_to_world)

    proj_matrix = proj_matrix.astype(float)
    pc = xyz_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3] # camera coordinate points
    uvw = pc @ proj_matrix[:, :3].T + proj_matrix[:, 3] # pixel coordinates


    z = uvw[:, 2]
    u = uvw[:, 0] / z
    v = uvw[:, 1] / z

    valid_pts = (z > min_depth) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    idx = np.where(valid_pts)[0]
    return u[idx], v[idx], z[idx], idx
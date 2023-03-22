import os
import pickle
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rt
import open3d as o3d


def get_color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def shift_mask(mask, ZY):
    if ZY == "ZY20210800001":
        dx, dy = 0, 0
    elif (ZY == "ZY20210800002") or (ZY == "ZY20210800004"):
        dx, dy = 15, -3
    elif ZY == "ZY20210800003":
        dx, dy = 15, -30
    else:
        raise NotImplementedError

    rows, cols, _ = mask.shape
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(mask, MAT, (cols, rows))
    return dst


def read_pose_from_json(pose_path, num=0):
    raw_data = json.load(open(pose_path, "r"))
    if "dataList" in raw_data:
        raw_pose = raw_data["dataList"][num]
    else:
        raw_pose = raw_data["objects"][num]
    
    translation, rotation, scale = raw_pose["center"], raw_pose["rotation"], raw_pose["dimensions"]
    translation = np.float32([translation["x"], translation["y"], translation["z"]])
    rotation = np.float32([rotation["x"], rotation["y"], rotation["z"]])
    rotation = Rt.from_euler('XYZ', rotation).as_matrix()
    scale = np.float32([scale["length"], scale["width"], scale["height"]])

    pose = {
        "translation": translation.reshape(3, 1),
        "rotation": rotation,
        "scale": scale.reshape(3, 1),
    }
    return pose


def read_poses_from_pkl(poses_path):
    raw_poses = pickle.load(open(poses_path, "rb"))
    return raw_poses


def get_specific_semantic_mask(mask, label=0):
    color_map = get_color_map(N=10)
    c = color_map[label]
    specific_mask = (mask[..., 0] == c[0]) & (mask[..., 1] == c[1]) & (mask[..., 2] == c[2])
    return specific_mask


def read_object_pcd(video_dir, idx, sampling_ratio=0.1):
    # read data
    p1 = video_dir.find("ZY")
    p2 = video_dir[p1:].find("/")
    ZY = video_dir[p1:p1+p2]
    if ZY == "ZY20210800001":
        rgb_path = os.path.join(video_dir, "align_image", str(idx) + ".jpg")
    else:
        rgb_path = os.path.join(video_dir, "shift_rgb", str(idx) + ".jpg")
    depth_path = os.path.join(video_dir, "align_depth", str(idx) + ".png")
    mask_path = os.path.join(video_dir, "refine_2Dseg", "mask", str(idx).zfill(5) + ".png")  # unshifted
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # unshifted
    mask = shift_mask(mask, ZY)  # shifted

    # read intrinsic
    intrinsic_dir = "/share/datasets/HOI4D_intrinsics"
    intrinsic = np.load(os.path.join(intrinsic_dir, ZY, "intrin.npy"))  # shape = (3, 3)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    
    # process data
    object_mask = get_specific_semantic_mask(mask, label=1)
    object_depth = np.float32(depth)
    object_depth[~object_mask] = 0
    object_depth = o3d.geometry.Image(object_depth)
    object_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, object_depth, convert_rgb_to_intensity=False)
    object_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(object_rgbd, intrinsic, extrinsic=np.eye(4))
    object_pcd = object_pcd.voxel_down_sample(voxel_size=0.001)
    object_pcd, _ = object_pcd.remove_radius_outlier(nb_points=500, radius=0.03)
    object_pcd = object_pcd.random_down_sample(sampling_ratio=0.1)
    return object_pcd


def read_model_pcd(model_path, N_points=10000):
    model_mesh = o3d.io.read_triangle_mesh(model_path)
    model_pcd = model_mesh.sample_points_uniformly(number_of_points=N_points, seed=0)
    model_points = np.float32(model_pcd.points)
    center = (np.min(model_points, axis=0) + np.max(model_points, axis=0)) / 2
    model_points -= center
    return model_points


if __name__ == "__main__":
    video_dir = "/share/datasets/HOI4D_overall/ZY20210800001/H1/C12/N33/S165/s02/T1"
    idx = 0

    pose_path = os.path.join(video_dir, "objpose", str(idx) + ".json")
    if not os.path.isfile(pose_path):
        pose_path = os.path.join(video_dir, "objpose", str(idx).zfill(5) + ".json")
    if not os.path.isfile(pose_path):
        raise NotImplementedError
    gt_pose = read_pose_from_json(pose_path)
    object_pcd = read_object_pcd(video_dir, idx)

    object_points = np.float32(object_pcd.points)
    object_colors = np.float32(object_pcd.colors)
    object_points = (object_points - gt_pose["translation"].reshape(3)) @ gt_pose["rotation"]
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(object_points)
    object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    o3d.io.write_point_cloud("gt.ply", object_pcd)

import os
import numpy as np
import cv2
from process_data import shift_mask, get_specific_semantic_mask, read_pose_from_json


def get_video_names(dataset_dir, category):
    video_names = []
    for ZY in os.listdir(dataset_dir):
        p1 = os.path.join(dataset_dir, ZY)
        if not "ZY" in ZY:
            continue
        for H in os.listdir(p1):
            p2 = os.path.join(p1, H)
            if not "H" in H:
                continue
            p3 = os.path.join(p2, category)
            if not os.path.isdir(p3):
                continue
            for N in os.listdir(p3):
                p4 = os.path.join(p3, N)
                if not "N" in N:
                    continue
                for S in os.listdir(p4):
                    p5 = os.path.join(p4, S)
                    if not "S" in S:
                        continue
                    for s in os.listdir(p5):
                        p6 = os.path.join(p5, s)
                        if not "s" in s:
                            continue
                        for T in os.listdir(p6):
                            p7 = os.path.join(p6, T)
                            if not "T" in T:
                                continue
                            if ZY == "ZY20210800001":
                                if not os.path.isfile(os.path.join(p7, "align_image", "0.jpg")):
                                    continue
                            else:
                                if not os.path.isfile(os.path.join(p7, "shift_rgb", "0.jpg")):
                                    continue
                            flag = True
                            for i in range(300):
                                pose_path = os.path.join(p7, "objpose", str(i) + ".json")
                                if not os.path.isfile(pose_path):
                                    pose_path = os.path.join(p7, "objpose", str(i).zfill(5) + ".json")
                                if not os.path.isfile(pose_path):
                                    flag = False
                                    break
                            if not flag:
                                continue
                            video_names.append(os.path.join(ZY, H, category, N, S, s, T))
    return video_names


def prepare_BundleTrack_data(dataset_dir, HOI4D_Sim_dir, intrinsics_dir, video_name, save_dir):
    save_dir = os.path.join(save_dir, video_name.replace("/", "_"))
    rgb_dir = os.path.join(save_dir, "rgb")
    depth_dir = os.path.join(save_dir, "depth")
    masks_dir = os.path.join(save_dir, "masks")
    poses_dir = os.path.join(save_dir, "annotated_poses")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    p1 = video_name.find("ZY")
    p2 = video_name[p1:].find("/")
    ZY = video_name[p1:p1+p2]
    intrinsic = np.load(os.path.join(intrinsics_dir, ZY, "intrin.npy"))
    np.savetxt(os.path.join(save_dir, "cam_K.txt"), intrinsic)

    for i in range(300):
        '''
        # HOI4D_Sim -> HOI4D_BundleTrack
        rgb_path = os.path.join(HOI4D_Sim_dir, video_name, "rgb", str(i).zfill(5) + ".png")
        depth_path = os.path.join(HOI4D_Sim_dir, video_name, "depth", str(i).zfill(5) + ".png")
        mask_path = os.path.join(HOI4D_Sim_dir, video_name, "2Dseg", str(i).zfill(5) + ".png")
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_path, cv2.CV_16UC1)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        '''
        # HOI4D_overall -> HOI4D_BundleTrack
        if ZY == "ZY20210800001":
            rgb_path = os.path.join(dataset_dir, video_name, "align_image", str(i) + ".jpg")
        else:
            rgb_path = os.path.join(dataset_dir, video_name, "shift_rgb", str(i) + ".jpg")
        depth_path = os.path.join(dataset_dir, video_name, "align_depth", str(i) + ".png")
        mask_path = os.path.join(dataset_dir, video_name, "refine_2Dseg", "mask", str(i).zfill(5) + ".png")  # unshifted
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depth_path, cv2.CV_16UC1)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]  # unshifted
        mask = shift_mask(mask, ZY)  # shifted
        
        object_mask = get_specific_semantic_mask(mask, label=1).astype(np.uint8)

        cv2.imwrite(os.path.join(rgb_dir, str(i).zfill(4) + ".png"), rgb)
        cv2.imwrite(os.path.join(depth_dir, str(i).zfill(4) + ".png"), depth)
        cv2.imwrite(os.path.join(masks_dir, str(i).zfill(4) + ".png"), object_mask)

        pose_path = os.path.join(dataset_dir, video_name, "objpose", str(i) + ".json")
        if not os.path.isfile(pose_path):
            pose_path = os.path.join(dataset_dir, video_name, "objpose", str(i).zfill(5) + ".json")
        if not os.path.isfile(pose_path):
            raise NotImplementedError
        gt_pose = read_pose_from_json(pose_path)
        T = np.eye(4)
        T[:3, :3] = gt_pose["rotation"]
        T[:3, 3:] = gt_pose["translation"].reshape(3, 1)
        np.savetxt(os.path.join(poses_dir, str(i).zfill(4) + ".txt"), T)


if __name__ == "__main__":
    # HOI4D_Sim_dir = "/localdata_hdd1/HOI4D_Sim"
    HOI4D_Sim_dir = None

    ############ CHANGE THIS ############
    category = "C13"
    dataset_dir = "/share/datasets/HOI4D_overall"
    save_dir = os.path.join("/localdata_hdd1/HOI4D_BundleTrack", category)
    intrinsics_dir = "/share/datasets/HOI4D_intrinsics"
    ########################

    video_names = get_video_names(dataset_dir, category)

    wr = open(category + ".txt", "w")
    for vn in video_names:
        wr.write(vn + "\n")
    wr.close()

    print("len(video_names) = ", len(video_names))
    for video_name in video_names:
        prepare_BundleTrack_data(dataset_dir, HOI4D_Sim_dir, intrinsics_dir, video_name, save_dir)

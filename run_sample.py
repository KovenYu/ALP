import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
import json
import subprocess
import pickle
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='This must contain two sub-folders: masks and images')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for best pose assets. If None,'
                                                                 'will use args.input_dir/best_pose/')
parser.add_argument('--mesh_path', type=str, help='Path to the mesh used.')
parser.add_argument('--crop_size', type=int, default=1000, help='Image size')
parser.add_argument('--original_size', type=int, default=2000, help='Image size')
parser.add_argument('--focal_length', type=float, default=35, help='Focal ratio')
parser.add_argument('--n_rot', type=int, default=4, help='How many initial rotations along gravity-axis to try?')
parser.add_argument('--train_res', type=int, default=1000)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--n_iter', type=int, default=500)
parser.add_argument('--model_height', type=float, default=2, help='height of mesh model.')
parser.add_argument('--pose_lr', type=float, default=0.1)
parser.add_argument('--save_metalness', action='store_true', help='Save metalness for lighting estimation.')
parser.add_argument('--init_cam_dist', type=float, default=0, help='if not 0, use this as initial camera distance.')

args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = os.path.join(args.input_dir, 'best_pose')

# generate poses bounds
def gen_poses_bounds(output_dir, args):
    r = R.from_euler('z', 90, degrees=True)
    rotMat = r.as_matrix()

    try:
        focal_length = np.load(os.path.join(args.input_dir, 'focal.npy'))
        print('Using focal length = {} from focal.npy'.format(focal_length))
    except FileNotFoundError:
        focal_length = args.focal_length
        print('Using focal length = {} from args'.format(focal_length))
    try:
        crop_factor, crop_size, original_size = np.load(os.path.join(args.input_dir, 'crop_factor.npy'))
        print('Using crop_factor = {} from crop_factor.npy'.format(crop_factor))
    except FileNotFoundError:
        crop_size = args.crop_size
        original_size = args.original_size
    sensor_size = 23.8
    model_height = args.model_height
    effective_sensor_size = sensor_size * crop_size / original_size
    focal_ratio = focal_length / effective_sensor_size
    init_cam_dist = focal_ratio * model_height
    init_cam_dist = init_cam_dist if args.init_cam_dist == 0 else args.init_cam_dist
    print('focal_ratio = {}, model_height = {}, init_cam_dist = {}'.format(focal_ratio, model_height, init_cam_dist))

    trans = np.zeros((3, 1))
    trans[2, 0] = init_cam_dist

    intrinsics = np.zeros((3, 1))
    intrinsics[0:2, 0] = args.train_res
    intrinsics[2, 0] = focal_ratio * args.train_res

    projMat = np.column_stack((rotMat, trans, intrinsics)).reshape((-1))
    outMat = np.concatenate((projMat, np.array([0, 0]))).reshape((1, -1))
    np.save(os.path.join(output_dir, 'poses_bounds.npy'), outMat)
    np.save(os.path.join(output_dir, 'cam_intrinsics.npy'), np.array([effective_sensor_size, focal_length]))


# config for pose optimization for nvdiffrecmc
def gen_config(input_dir_llff, base_mesh, out_dir, rot_init=[1, 0, 0, 0], random_textures=False, camera_space_light=False):
    config = {
        "ref_mesh": input_dir_llff,
        "random_textures": random_textures,
        "iter": args.n_iter,
        "save_interval": 50,
        "texture_res": [ 2048, 2048 ],
        "train_res": [args.train_res, args.train_res],
        "val_res": [args.crop_size, args.crop_size],
        "batch": 1,
        "n_samples": args.n_samples,
        "learning_rate": [args.pose_lr, 0., 0.1],  # pose, material, light
        "dmtet_grid" : 64,
        "mesh_scale" : 3,
        "background" : "black",
        "display": [{"latlong" : True}, {"bsdf" : "kd"}, {"bsdf" : "ks"}, {"bsdf" : "normal"}],
        "out_dir": out_dir,
        "base_mesh": base_mesh,
        "lock_pos": False,
        "opt_cam_pos": True,
        "learn_light": True,
        "lock_light": False,
        "run_proposed": True,
        "no_perturbed_nrm": True,
        "rot_init": rot_init,
        "smooth_envmap": True,
        "smooth_envmap_wt": 0.01,
        "denoiser": "bilateral",
        "save_validation": False,
        "camera_space_light": camera_space_light,
        "no_texture_opt": True
    }
    return config


# input to folder for training example -- folder must contain two sub-folders: masks and images
def main(input_folder, base_mesh, args):
    gen_poses_bounds(input_folder, args)

    #  0, 90, 180 and 270 degree rotation around y axis
    # rot_init_list = [[1.0, 0.0, 0.0, 0.0], [0.707, 0.0,  0.707, 0.0], [0.0, 0.0, 1.0, 0.0], [-0.707, 0,  0.707, 0]]
    total_opening_angle = 180
    start = -total_opening_angle / 2
    offset = total_opening_angle / (args.n_rot + 1)
    interval = offset
    for i in range(args.n_rot):
        # r = R.from_euler('yx', [360/args.n_rot*i, -30], degrees=True).as_quat()  # (x, y, z, w)
        r = R.from_euler('yx', [start + offset + interval*i, 0], degrees=True).as_quat()  # (x, y, z, w)
        rot_init = [r[3], r[0], r[1], r[2]]

        config = gen_config(input_folder, base_mesh, os.path.join(input_folder, str(i)), rot_init)
        config["save_validation"] = True

        config_path = os.path.join(input_folder, "config%d.json" % (i))
        with open(config_path, "w") as outfile:
            json.dump(config, outfile)
        result = subprocess.run(["python", "train.py", "--config", config_path])
        if result.returncode != 0:
            raise Exception("Error in subprocess")

        # finding best pose
    best_psnr = -1
    best_idx = -1
    for i in range(args.n_rot):
        val_path = os.path.join(input_folder, str(i), "val.pkl")
        with open(val_path, 'rb') as f:
            psnr = pickle.load(f)
            if psnr > best_psnr:
                best_psnr = psnr
                best_idx = i
    assert best_idx >= 0


    best_pose_dir =  os.path.join(input_folder, str(best_idx), "validate")
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(os.path.join(best_pose_dir, "kd.exr"), os.path.join(args.output_dir, "albedo.exr"))
    shutil.copy(os.path.join(best_pose_dir, "ks_r.exr"), os.path.join(args.output_dir, "roughness.exr"))
    if args.save_metalness:
        shutil.copy(os.path.join(best_pose_dir, "ks_m.exr"), os.path.join(args.output_dir, "metalness.exr"))
    shutil.copy(os.path.join(best_pose_dir, "ks_o.exr"), os.path.join(args.output_dir, "occlusion.exr"))
    shutil.copy(os.path.join(best_pose_dir, "normal.exr"), os.path.join(args.output_dir, "normal.exr"))
    shutil.copy(os.path.join(best_pose_dir, "mask.exr"), os.path.join(args.output_dir, "asset_mask.exr"))
    shutil.copy(os.path.join(input_folder, "masks/0.png"), os.path.join(args.output_dir, "ref_mask.png"))
    shutil.copy(os.path.join(input_folder, "cam_intrinsics.npy"), os.path.join(args.output_dir, "cam_intrinsics.npy"))


# input to folder for training example -- folder must contain two sub-folders: masks and images
def single_pose_opt(input_folder, base_mesh):

    rot_init_list = [1.0, 0.0, 0.0, 0.0]

    # running pose optimization for each rot init
    config = gen_config(input_folder, base_mesh, os.path.join(input_folder, 'aligned'), rot_init_list, True)
    config["save_validation"] = True

    config_path = os.path.join(input_folder, "config.json")
    with open(config_path, "w") as outfile:
        json.dump(config, outfile)
    subprocess.run(["python", "train.py", "--config", config_path])




if __name__=="__main__":
    try:
        main(args.input_dir, args.mesh_path, args)
    except:
        np.save(os.path.join(args.input_dir, "error.npy"), 0)
        import sys
        sys.exit(1)
    # main('/viscam/projects/alp/2022_ALP/nvdiffrecmc/data/nerd/quant_test1', '/viscam/u/samirag/nvdiffrec/archive/out/lightbox2_half_ks/mesh/mesh.obj')
    # main('/viscam/projects/alp/2022_ALP/nvdiffrecmc/data/nerd/quant_test2', '/viscam/projects/alp/2022_ALP/nvdiffrecmc/data/diet/lightbox2_full_roughness/mesh/mesh.obj')
    # main('/viscam/projects/alp/2022_ALP/nvdiffrecmc/data/nerd/quant_test3', '/viscam/u/samirag/nvdiffrec/archive/out/lightbox2_half_roughness/mesh/mesh.obj')
    # main('/viscam/u/samirag/nvdiffrecmc/data/nerd/quant_test4', '/viscam/u/samirag/nvdiffrec/archive/out/1017_coketm_kskd_v2/mesh/mesh.obj')
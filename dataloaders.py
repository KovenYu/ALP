import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import hdrio, os
import numpy as np
from helpers import find_all_output_dirs
import cv2


class RealCompletionDataset(Dataset):
    def __init__(self, dataset_dir, target_depth=0, load_height=256):
        super(RealCompletionDataset, self).__init__()
        self.sample_dirs = find_all_output_dirs(dataset_dir, 0, target_depth)
        self.eps = 1e-6
        self.load_height = load_height

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        emap_optim = hdrio.imread(os.path.join(sample_dir, 'emap_learned.exr'))[..., :3]
        emap_optim = emap_optim.clip(min=0)
        target_mean = 0.5
        scale = target_mean / emap_optim.mean()
        emap_optim = emap_optim * scale
        _, emap_optim_log = self.transform_emap(emap_optim)  # [3, H, W]
        mean, std = emap_optim_log.mean(dim=[1, 2], keepdim=True), emap_optim_log.std(dim=[1, 2], keepdim=True)  # [3, 1, 1]
        emap_conf = hdrio.imread(os.path.join(sample_dir, 'confidence_map.exr'))  # [H, W, 1]
        _, emap_conf_log = self.transform_emap(emap_conf)
        ret_dict = {'emap_optim_log': emap_optim_log,
                    'optim_mean': mean,
                    'optim_std': std,
                    'emap_conf_log': emap_conf_log,
                    'scale': scale,
                    }
        return ret_dict

    def transform_emap(self, emap):
        """
        transform to log space
        :param emap: [H, W, 3] np.ndarray
        :return:
            emap_transformed: [3, H, W], in log space
        """
        emap_tensor = torch.tensor(emap).permute([2, 0, 1])  # [3, H, W]
        if emap_tensor.shape[1] != self.load_height:
            h, w = self.load_height, self.load_height*2
            emap_tensor = torch.nn.functional.interpolate(emap_tensor[None, ...], [h, w], mode='bilinear', align_corners=False).squeeze(0)
        emap_log = (emap_tensor + self.eps).log10()
        return emap_tensor, emap_log


class CompletionDataset(Dataset):
    def __init__(self, dataset_dir, load_height=256):
        super(CompletionDataset, self).__init__()
        load_dir = dataset_dir
        self.sample_dirs = find_all_output_dirs(load_dir, 0, 1)
        # filter out sample_dirs whose basename does not start with a number
        self.sample_dirs = [d for d in self.sample_dirs if d.split('/')[-1][0].isdigit()]
        debug = False
        if debug:
            self.sample_dirs = self.sample_dirs[:16]
        self.eps = 1e-6
        self.load_height = load_height

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        emap_optim = hdrio.imread(os.path.join(sample_dir, 'emap_learned.exr'))[..., :3]  # already mean~=0.5
        emap_gt = hdrio.imread(os.path.join(sample_dir, 'emap.exr'))[..., :3]  # already mean~=0.5
        _, emap_optim_log = self.transform_emap(emap_optim)  # [3, H, W]
        mean, std = emap_optim_log.mean(dim=[1, 2], keepdim=True), emap_optim_log.std(dim=[1, 2], keepdim=True)  # [3, 1, 1]
        emap_gt_tensor, emap_gt_log = self.transform_emap(emap_gt)
        emap_valid = emap_gt_tensor.mean(dim=0, keepdim=True) >= self.eps
        emap_conf = hdrio.imread(os.path.join(sample_dir, 'confidence_map.exr'))  # [H, W, 1]
        _, emap_conf_log = self.transform_emap(emap_conf)
        ret_dict = {'emap_optim_log': emap_optim_log,
                    'emap_gt_log': emap_gt_log,
                    'emap_valid': emap_valid,
                    'emap_conf_log': emap_conf_log,
                    'optim_mean': mean,
                    'optim_std': std,
                    }
        return ret_dict

    def transform_emap(self, emap):
        """
        transform to log space
        :param emap: [H, W, 3] np.ndarray
        :return:
            emap_transformed: [3, H, W], in log space
        """
        emap_tensor = torch.tensor(emap).permute([2, 0, 1])  # [3, H, W]
        if emap_tensor.shape[1] != self.load_height:
            h, w = self.load_height, self.load_height*2
            emap_tensor = torch.nn.functional.interpolate(emap_tensor[None, ...], [h, w], mode='bilinear', align_corners=False).squeeze(0)
        emap_log = (emap_tensor + self.eps).log10()
        return emap_tensor, emap_log


class SingleImageDataset(Dataset):
    def __init__(self, input_dir, focal_ratio=50/23.8, default_rough=0.15, default_metal=1.,
                 shifted_emap=False, repr_emap_in_cam=False,
                 repr_emap_for_real_world=False, use_alpha_channel_over_mask=False):
        """
        :param input_dir:
        :param focal_ratio: (focal_x / sensor_width), assuming this is the same as (focal_y / sensor_height)
        :param default_rough: float
        :param default_metal: float
        :param shifted_emap: Set to True if you shifted your GT emap to the left before loading it in Blender
        for synthesizing your data.
        :param repr_emap_in_cam: If True, emap frame is defined the same as camera frame. There is no "world".
        This is what happens for real images.
        :param repr_emap_for_real_world: (This has been messy, to refactor this if-else chain later)
        if True, we simply transform rays and normals by a cam2world matrix named 'cam2realworld.npy',
        so that we can optimize a envmap in the "realworld" frame.
        This is a temporary design for material accquisition.
        """
        gt_img_path = os.path.join(input_dir, 'hdr.exr')
        gt_img = hdrio.imread(gt_img_path)  # [H, W, 4]
        if use_alpha_channel_over_mask:
            ref_mask = gt_img[..., 3]
        else:
            try:
                ref_mask_path = gt_img_path.replace('hdr.exr', 'ref_mask.png')
                ref_mask = hdrio.imread(ref_mask_path)  # [H, W, 1 or 3 or None]
            except:
                try:
                    ref_mask_path = gt_img_path.replace('hdr.exr', 'ref_mask.exr')
                    ref_mask = hdrio.imread(ref_mask_path)  # [H, W, 1 or 3 or None]
                except:
                    ref_mask = gt_img[..., 3]
        if len(ref_mask.shape) == 3:
            ref_mask = ref_mask.mean(axis=2, keepdims=True)
        else:
            ref_mask = ref_mask[..., None]

        try:
            asset_mask_path = gt_img_path.replace('hdr.exr', 'asset_mask.exr')
            asset_mask = hdrio.imread(asset_mask_path)  # [H, W, 1 or 3]
        except FileNotFoundError:
            asset_mask_path = gt_img_path.replace('hdr.exr', 'ldr.png')
            asset_mask = hdrio.imread(asset_mask_path)  # [H, W, 4]
            asset_mask = asset_mask[..., 3:4]
        if len(asset_mask.shape) == 3:
            asset_mask = asset_mask.mean(axis=2, keepdims=True)
        else:
            asset_mask = asset_mask[..., None]

        # resize ref_mask to the size of asset_mask by cv2
        ref_mask = cv2.resize(ref_mask, (asset_mask.shape[1], asset_mask.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]
        alpha = torch.tensor(ref_mask) * torch.tensor(asset_mask)  # [H, W, 1]

        try:
            gt_albedo_path = gt_img_path.replace('hdr.exr', 'albedo.exr')
            gt_albedo = hdrio.imread(gt_albedo_path)  # [H, W, 4]
        except FileNotFoundError:
            gt_albedo_path = gt_img_path.replace('hdr.exr', 'albedo.png')
            gt_albedo = hdrio.imread(gt_albedo_path)  # [H, W, 4]
            print('Loading albedo from .PNG, consider replacing it with .EXR to avoid any tone mapping inconsistency.')
        gt_albedo = torch.tensor(gt_albedo[..., :3]) * alpha  # [H, W, 3]

        gt_img_rgb = torch.tensor(gt_img[..., :3]) * alpha  # [H, W, 3]
        gt_normal_path = gt_img_path.replace('hdr.exr', 'normal.exr')
        gt_normal_map = hdrio.imread(gt_normal_path)  # [H, W, 3], in camera frame
        gt_normal_map = torch.tensor(gt_normal_map)  # [H, W, 3]
        gt_normal = F.normalize(gt_normal_map * 2 - 1, dim=-1)

        try:
            gt_roughness_path = gt_img_path.replace('hdr.exr', 'roughness.exr')
            gt_roughness = hdrio.imread(gt_roughness_path)  # [H, W, 1]
            gt_roughness = gt_roughness[..., 0:1] * gt_roughness[..., -1:]
            gt_roughness = torch.tensor(gt_roughness)  # [H, W, 1]
            print('Loaded roughness map.')
        except IOError:
            gt_roughness = torch.ones_like(alpha) * default_rough
            print('Using default roughness = {}'.format(default_rough))

        try:
            gt_metalness_path = gt_img_path.replace('hdr.exr', 'metalness.exr')
            gt_metalness = hdrio.imread(gt_metalness_path)  # [H, W, 1]
            gt_metalness = gt_metalness[..., 0:1] * gt_metalness[..., -1:]
            gt_metalness = torch.tensor(gt_metalness)  # [H, W, 1]
            print('Loaded metalness map.')
        except IOError:
            gt_metalness = torch.ones_like(alpha) * default_metal
            print('Using default metalness = {}'.format(default_metal))

        try:
            occlusion_path = gt_img_path.replace('hdr.exr', 'occlusion.exr')
            occlusion = hdrio.imread(occlusion_path)  # [H, W, 1]
            occlusion = torch.tensor(occlusion)  # [H, W, 1]
            print('Loaded occlusion map.')
        except IOError:
            occlusion = torch.zeros_like(alpha)
            print('Did not load occlusion map. Assuming no ambient occlusion.')

        H, W, _ = gt_img.shape

        try:
            cam_intrinsics_path = os.path.join(input_dir, 'cam_intrinsics.npy')
            cam_intrinsics = np.load(cam_intrinsics_path)
            focal_ratio = float(cam_intrinsics[1] / cam_intrinsics[0])
            print('Loaded camera intrinsics = {:.2f}/{:.2f}.'.format(cam_intrinsics[1], cam_intrinsics[0]))
        except FileNotFoundError:
            print('Using default focal ratio = 50 / 23.8.')

        ray_dir = - construct_ray_dir(focal_ratio, H, W)  # [H, W, 3], openGL format, negative for surface shading

        repr_emap_in_world = not repr_emap_in_cam
        if repr_emap_in_world:
            repr_emap_for_blender = not repr_emap_for_real_world
            if repr_emap_for_blender:  # For comparison to GT emap defined in world frame.
                """ When blender loads an envmap, it takes envmap's center point as +X, which is -Z in cam/envmap frame. 
                    Here 'emap' refers to the envmap frame, defined by the figure in skylib repo. """
                cam2Bworld = torch.tensor(np.load(gt_img_path.replace('hdr.exr', 'cam2world.npy')))  # [4, 4]
                cam2Bworld_R = cam2Bworld[:3, :3]  # Blender world coord
                print('Loaded cam2world matrix.')
                emap2Bworld_R = torch.tensor([[0., 0, -1],
                                              [-1, 0, 0],
                                              [0, 1, 0]])
                """ However, we don't like this Blender envmap frame, and we instead want the center point to be +Y.
                    We can't modify how Blender defines its envmap frame, but we instead shift envmap before loading it to Blender,
                    whiel we keep GT envmap as the unshifted version. Effectively, this changes envmap frame to what we want. """
                if shifted_emap:
                    emap2Bworld_R = torch.tensor([[1., 0., 0.],
                                                  [0., 0., -1.],
                                                  [0., 1., 0.]])
                cam2emap = torch.matmul(emap2Bworld_R.inverse(), cam2Bworld_R)  # [3, 3]
                ray_dir = torch.matmul(cam2emap[None, None, ...], ray_dir[..., None]).squeeze(-1)  # [H, W, 3]
                gt_normal = torch.matmul(cam2emap[None, ...], gt_normal[..., None]).squeeze(-1)  # [H, W, 3]
            else:
                cam2world = torch.tensor(np.load(gt_img_path.replace('hdr.exr', 'cam2realworld.npy')))  # [4, 4]
                cam2world = cam2world[:3, :3]
                ray_dir = torch.matmul(cam2world[None, None, ...], ray_dir[..., None]).squeeze(-1)  # [H, W, 3]
                gt_normal = torch.matmul(cam2world[None, ...], gt_normal[..., None]).squeeze(-1)  # [H, W, 3]

        try:
            concave_map_path = gt_img_path.replace('hdr.exr', 'concave_map.png')
            concave_map = hdrio.imread(concave_map_path)  # [H, W]
            concave_map = torch.tensor(concave_map)[..., None]  # [H, W, 1]
            print('Loaded concave map.')
        except FileNotFoundError:
            print('No concave map used.')
            concave_map = torch.zeros_like(alpha)
        confidence_map = 1 - concave_map  # [H, W, 1]
        confidence_albedo = gt_albedo.mean(dim=-1, keepdim=True) > 0.7  # black albedo means unconfident; 0.1 because anti-alias
        # confidence_map = confidence_map * confidence_albedo

        H_idx, W_idx = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        hw_idx = torch.stack([H_idx.flatten(), W_idx.flatten()], dim=-1)  # [P, 2]

        self.H, self.W = H, W
        self.ray_dir = ray_dir  # [H, W, 3]
        self.alpha_map = alpha  # [H, W, 1]
        self.img = gt_img_rgb  # [H, W, 3]
        self.normal = gt_normal  # [H, W, 3]
        self.albedo = gt_albedo  # [H, W, 3]
        self.roughness = gt_roughness
        self.metalness = gt_metalness
        self.occlusion = occlusion
        self.hw_idx = hw_idx  # [HxW, 2]
        self.flat_idx = torch.arange(H*W)[:, None]  # [HxW, 1]
        self.confidence_map = confidence_map  # [H, W, 1]

        viz_px_mask = alpha.squeeze() >= 1e-4  # [H, W, 1]
        non_zero = viz_px_mask.nonzero()
        h_min, h_max, w_min, w_max = non_zero[:, 0].min(), non_zero[:, 0].max(), non_zero[:, 1].min(), non_zero[:, 1].max()
        self.v_min, self.v_max, self.u_min, self.u_max = h_min/H, h_max/H, w_min/W, w_max/W

        viz_px_mask = viz_px_mask.flatten()
        self.ray_dir_viz = self.ray_dir.flatten(0, 1)[viz_px_mask]  # [P, 3]
        self.alpha_map_viz = alpha.flatten(0, 1)[viz_px_mask]  # [P, 1]
        self.img_viz = gt_img_rgb.flatten(0, 1)[viz_px_mask]  # [P, 3]
        self.normal_viz = gt_normal.flatten(0, 1)[viz_px_mask]  # [P, 3]
        self.albedo_viz = gt_albedo.flatten(0, 1)[viz_px_mask]  # [P, 3]
        self.roughness_viz = gt_roughness.flatten(0, 1)[viz_px_mask]
        self.metalness_viz = gt_metalness.flatten(0, 1)[viz_px_mask]
        self.occlusion_viz = occlusion.flatten(0, 1)[viz_px_mask]
        self.hw_idx_viz = hw_idx[viz_px_mask]  # [P, 2]
        self.flat_idx_viz = self.flat_idx[viz_px_mask]  # [P, 1]

    def __len__(self):
        pass

    def __getitem__(self, idx):
        """Pytorch Dataloader seems to be very slow; manual indexing is much faster. Not sure why"""
        pass

    def get_rgb_map(self):
        return self.img.reshape([self.H, self.W, 3])

    def remap_uv(self, uv_):
        """
        :param uv_: [P, 2]
        :return:
            uv: [P, 2]
        """
        u_, v_ = uv_[:, 0], uv_[:, 1]
        u = self.u_min + u_ * (self.u_max - self.u_min)
        v = self.v_min + v_ * (self.v_max - self.v_min)
        uv = torch.stack([u, v], dim=-1)
        return uv

    def sample_uv(self, uv_):
        """
        :param uv_: [P, 2], ranging [0, 1]
        :return:
            batch_dict:
        """
        visible_uv = self.remap_uv(uv_)  # [P, 2]

        def grid_sample_uv(input_map, uv, normalize=False):
            """
            :param input_map: [H, W, C]
            :param uv: [P, 2], ranging [0, 1]
            :param normalize:
            :return:
                sampled: [P, C]
            """
            input_map_ = input_map[None, ...].permute([0, 3, 1, 2])  # [1, C, H, W]
            grid = uv * 2 - 1
            grid = grid[None, None, ...]  # [1, 1, P, 2]
            sampled_map = F.grid_sample(input_map_, grid, padding_mode='border', align_corners=True)  # [1, C, 1, P]
            sampled = sampled_map.squeeze(0).squeeze(1).permute([1, 0])  # [P, C]
            if normalize:
                sampled = sampled / (sampled.norm(dim=-1, keepdim=True) + 1e-8)
            return sampled

        alpha_batch = grid_sample_uv(self.alpha_map, visible_uv)
        albedo_batch = grid_sample_uv(self.albedo, visible_uv)
        if self.roughness is None:
            roughness_batch = None
        else:
            roughness_batch = grid_sample_uv(self.roughness, visible_uv)
        metalness_batch = grid_sample_uv(self.metalness, visible_uv)
        occlusion_batch = grid_sample_uv(self.occlusion, visible_uv)
        normal_batch = grid_sample_uv(self.normal, visible_uv, normalize=True)
        is_abnormal_norm = normal_batch.norm(dim=-1) < 0.95
        normal_batch[is_abnormal_norm] = torch.tensor([0., 0., 1.])
        ray_dir_batch = grid_sample_uv(self.ray_dir, visible_uv, normalize=True)
        rgb_batch = grid_sample_uv(self.img, visible_uv)
        confidence_batch = grid_sample_uv(self.confidence_map, visible_uv)

        batch_dict = {'alpha_batch': alpha_batch,
                      'albedo_batch': albedo_batch,
                      'roughness_batch': roughness_batch,
                      'metalness_batch': metalness_batch,
                      'occlusion_batch': occlusion_batch,
                      'normal_batch': normal_batch,
                      'ray_dir_batch': ray_dir_batch,
                      'rgb_batch': rgb_batch,
                      'confidence_batch': confidence_batch,
                      }
        return batch_dict


def construct_ray_dir(focal_ratio, H, W):
    """
    :param focal_ratio: (focal_x / width); assuming this is the same as (focal_y / height)
    :param H:
    :param W:
    :return:
        ray_dir: [H, W, 3], normalized openGL format ray directions
    """
    ray_origin = torch.tensor([0., 0., 2 * focal_ratio])[None, None, :]  # [1, 1, 3]

    X_coord, Y_coord = (torch.arange(W) + 0.5) / W, (torch.arange(H) + 0.5) / H
    X_coord, Y_coord = X_coord * 2 - 1, - (Y_coord * 2 - 1)  # now X right, Y up
    X_coord, Y_coord = torch.meshgrid([X_coord, Y_coord], indexing='xy')  # [H, W], [H, W]
    pixel_coord = torch.stack([X_coord, Y_coord, torch.zeros_like(X_coord)], dim=-1)  # [H, W, 3]
    ray_dir_unnorm = pixel_coord - ray_origin
    ray_dir = F.normalize(ray_dir_unnorm, dim=-1)  # [H, W, 3]
    return ray_dir


def test_construct_ray_dir():
    focal_ratio = 1
    H = W = 2
    ray_dir = construct_ray_dir(focal_ratio, H, W)
    breakpoint()
    print(ray_dir)


if __name__ == '__main__':
    # test_construct_ray_dir()
    pass

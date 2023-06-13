import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import hdrio
import os
import argparse
from dataloaders import SingleImageDataset
import time, sys
import numpy as np
from helpers import Logger, LightWrapper

parser = argparse.ArgumentParser()

# IO options
parser.add_argument('--gt_envmap_path', default='data/brown_photostudio_02_2k_shifted.exr')
parser.add_argument('--input_dir', default='outputs/exp220801_render_cokecan/ortho_cokecan_nosv_nosp_nodiff_norough_notex',
                    help="should have input hdr/normal/albedo in it; output also goes here")
parser.add_argument('--roughness', default=1., type=float,
                    help="assuming no SV roughness")
parser.add_argument('--metalness', default=1., type=float,
                    help="default value; assuming no SV")
parser.add_argument('--emap_height', default=512, type=int,
                    help="emap resolution to be optimized, assuming width = 2 * height")
parser.add_argument('--batch_size', default=600**2, type=int)
parser.add_argument('--render_gt', action='store_true',
                    help="To render with GT envmap; do not do optimization.")
parser.add_argument('--emap_init', default='zeros',
                    help="Choose from {'zeros', 'gt_mean'}.")
parser.add_argument('--shifted_emap', action='store_true',
                    help="Did you shift emap to the left before loading in order to put camera to face Blender +Y?")
parser.add_argument('--repr_emap_in_cam', action='store_true',
                    help="For real images, there is no `world` frame, but only camera frame. Set this to True.")
parser.add_argument('--repr_emap_for_real_world', action='store_true',
                    help="For real images during material accquision, we want to represent envmap in world frame.")
parser.add_argument('--n_iters', default=1000, type=int)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--n_samples', default=600, type=int)
parser.add_argument('--log_space', action='store_true', help='Optimize envmap in its log space?')
parser.add_argument('--no_occlusion_map', action='store_true', help='Use occlusion map?')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--smooth_reg_weight', default=0.01, type=float)
parser.add_argument('--is_emap', action='store_true', help='if is emap, turn off smooth reg after 100 iters.')
parser.add_argument('--bad_pixel_conf_threshold', default=0.01, type=float, help='Fix pixel if confidence is below threshold.')
parser.add_argument('--use_alpha_channel_over_mask', action='store_true', help='If True, use target alpha channel as ref_mask.')
parser.add_argument('--emap_scale', default=1, type=float)


def main():
    args = parser.parse_args()
    input_dir = args.input_dir
    try:
        optimize_envmap(args, input_dir)
    except Exception as e:
        print(e)
        np.save(os.path.join(input_dir, 'error.npy'), 0)
        import sys
        sys.exit(1)


def optimize_envmap(args, input_dir):
    logger = Logger(os.path.join(input_dir, 'logs'))
    render_gt = args.render_gt

    emap_frequency = torch.zeros([1, 1, args.emap_height, 2*args.emap_height]).requires_grad_()

    # IBL on Principled BSDF
    n_samples = args.n_samples
    emap_tensor_ = torch.zeros([1, 3, args.emap_height, 2*args.emap_height]).requires_grad_()  # [1, 3, H_e, W_e]
    light_wrapper_net = LightWrapper(emap_tensor_, args.emap_height, n_samples, emap_init=args.emap_init, log_space=args.log_space,
                                     render_gt=render_gt)
    try:
        image_dataset = SingleImageDataset(input_dir, default_rough=args.roughness, default_metal=args.metalness, shifted_emap=args.shifted_emap,
                                           repr_emap_in_cam=args.repr_emap_in_cam, repr_emap_for_real_world=args.repr_emap_for_real_world,
                                           use_alpha_channel_over_mask=args.use_alpha_channel_over_mask)
    except Exception as e:
        print(e)
        emap_learned_np = light_wrapper_net.emap_learned.detach().squeeze().permute([1, 2, 0]).cpu().numpy()
        hdrio.imwrite(emap_learned_np, input_dir + '/emap_learned.exr')
        hdrio.imwrite(emap_learned_np, input_dir + '/emap_learned.hdr')
        hdrio.imwrite(emap_learned_np, input_dir + '/emap_learned.png')
        return -1
    if torch.cuda.device_count() > 1:
        light_wrapper_net = torch.nn.DataParallel(light_wrapper_net)
    optimizer = Adam(light_wrapper_net.parameters(), lr=args.lr, betas=(0.5, 0.5))
    optimizer.param_groups[0]['capturable'] = True  # a Pytorch 1.12 bug
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=args.patience)
    criterion = torch.nn.HuberLoss(reduction='none').cuda()

    P = args.batch_size
    n_iters = 1 if render_gt else args.n_iters
    for it in range(n_iters):
        if args.is_emap and it == 100:
            args.smooth_reg_weight = 0.
        t0 = time.time()
        uv_ = torch.rand(P, 2)
        batch_dict = image_dataset.sample_uv(uv_)
        alpha_batch = batch_dict['alpha_batch']
        albedo_batch = batch_dict['albedo_batch']
        metalness_batch = batch_dict['metalness_batch']
        roughness_batch = batch_dict['roughness_batch']
        roughness_batch = roughness_batch.clamp(0.01, 1.0)
        occlusion_batch = batch_dict['occlusion_batch']
        normal_batch = batch_dict['normal_batch']
        gt_rgb_batch = batch_dict['rgb_batch']
        view_dir_batch = batch_dict['ray_dir_batch']
        confidence_batch = batch_dict['confidence_batch']
        shaded_batch, emap_imp_uv_grid = light_wrapper_net(view_dir_batch, normal_batch, albedo_batch, roughness_batch,
                                                           metalness_batch)
        occlusion_batch = occlusion_batch if not args.no_occlusion_map else torch.zeros_like(occlusion_batch)
        shaded_batch = shaded_batch * alpha_batch * (1-occlusion_batch)

        if it < 10:  # Since we have a large number of samples, we only need a few batches to converge.
            emap_imp_uv_grid_nopar = emap_imp_uv_grid.flatten(0, 1)[None, ...]  # [N_gpu, P/N_gpu, S, 2] -> [1, P, S, 2]
            freq = torch.nn.functional.grid_sample(emap_frequency, emap_imp_uv_grid_nopar * 2 - 1, padding_mode='border', align_corners=False)
            hard_viz_batch = alpha_batch >= 0.5
            freq = freq.squeeze()[hard_viz_batch.squeeze()]
            freq_objective = (freq - 1).sum()
            freq_objective.backward()

        if not render_gt:
            loss = (criterion(shaded_batch, gt_rgb_batch) * confidence_batch).mean()
            # loss = (((tone_map(shaded_batch) - gt_rgb_batch)**2) * confidence_batch).mean()
            sampled_light, sampled_light_near = light_wrapper_net.sample_light_grad(10000)
            smooth_reg = criterion(sampled_light, sampled_light_near).mean()
            loss_ = loss + smooth_reg * args.smooth_reg_weight
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            try:
                light_wrapper_net.clamp_emap()
            except AttributeError:
                light_wrapper_net.module.clamp_emap()
            scheduler.step(loss.cpu())
            if it % 10 == 0:
                logger.print_log('it: {:04d}, loss: {:05f}, reg: {:05f}'.format(it, loss.item(), smooth_reg.item()))
            min_lr = args.lr / 1e2
            final_loss = loss.detach().cpu().item()
            if optimizer.param_groups[0]['lr'] <= min_lr:
                logger.print_log('LR smaller than {}, break optimization loop.'.format(min_lr))
                break

    with torch.no_grad():
        shaded_px = torch.zeros(image_dataset.H * image_dataset.W, 3)
        alpha = image_dataset.alpha_map.flatten(0, 1)  # [P, 1]
        albedo = image_dataset.albedo.flatten(0, 1)  # [P, 3]
        roughness = image_dataset.roughness.flatten(0, 1)
        metalness = image_dataset.metalness.flatten(0, 1)
        normal = image_dataset.normal.flatten(0, 1)
        view_dir = image_dataset.ray_dir.flatten(0, 1)
        n_batches = np.ceil(alpha.shape[0] / args.batch_size).astype(int)
        for i in range(n_batches):
            albedo_batch = albedo[i * P: (i + 1) * P]
            roughness_batch = roughness[i * P: (i + 1) * P]
            metalness_batch = metalness[i * P: (i + 1) * P]
            normal_batch = normal[i * P: (i + 1) * P]
            view_dir_batch = view_dir[i * P: (i + 1) * P]
            shaded_batch, _ = light_wrapper_net(view_dir_batch, normal_batch, albedo_batch, roughness_batch, metalness_batch)
            shaded_px[i * P: (i + 1) * P] = shaded_batch
        shaded_px = (shaded_px * alpha).reshape([image_dataset.H, image_dataset.W, 3])

    shaded_img = shaded_px.reshape([image_dataset.H, image_dataset.W, 3])
    shaded_savename = '/shaded_using_gt_envmap.exr' if render_gt else '/shaded_test.exr'
    hdrio.imwrite(shaded_img.detach().cpu().numpy(), input_dir + shaded_savename)
    hdrio.imwrite(shaded_img.detach().cpu().numpy(), input_dir + shaded_savename.replace('.exr', '.png'))
    try:
        emap_learned_np = light_wrapper_net.module.emap_learned.detach().squeeze().permute([1, 2, 0]).cpu().numpy()
    except AttributeError:
        emap_learned_np = light_wrapper_net.emap_learned.detach().squeeze().permute([1, 2, 0]).cpu().numpy()
    if args.log_space:
        emap_learned_np = 10 ** emap_learned_np

    emap_frequency_ = emap_frequency.grad
    emap_frequency_ = emap_frequency_.squeeze(dim=0).permute([1, 2, 0])
    emap_confidence = emap_frequency_ / emap_frequency_.max()
    if args.bad_pixel_conf_threshold > 0:
        bad_pixels_mask = emap_confidence[..., 0] < args.bad_pixel_conf_threshold  # [H, W]
        good_pixels_mask = ~bad_pixels_mask
        bad_pixels, neighbor_good_pixels = find_good_neighbors(bad_pixels_mask, good_pixels_mask, topk=10)  # [n_bad_pixels, 2], [n_bad_pixels, topk, 2]
        for bad_pixel_loc, neighbor_good_pixel_loc in zip(bad_pixels, neighbor_good_pixels):
            emap_learned_np[bad_pixel_loc[0], bad_pixel_loc[1]] = emap_learned_np[neighbor_good_pixel_loc[:, 0], neighbor_good_pixel_loc[:, 1]].mean(axis=0)

    hdrio.imwrite(emap_confidence.cpu().numpy(), input_dir + '/confidence_map.exr')
    hdrio.imwrite(emap_confidence.cpu().numpy()[:, :, 0], input_dir + '/confidence_map.png')

    hdrio.imwrite(emap_learned_np*args.emap_scale, input_dir + '/emap_learned.exr')
    hdrio.imwrite(emap_learned_np*args.emap_scale, input_dir + '/emap_learned.hdr')
    hdrio.imwrite(emap_learned_np*args.emap_scale, input_dir + '/emap_learned.png')

    hdrio.imwrite(image_dataset.get_rgb_map().detach().cpu().numpy(), input_dir + '/shaded_gt.exr')
    hdrio.imwrite(image_dataset.get_rgb_map().detach().cpu().numpy(), input_dir + '/shaded_gt.png')
    return final_loss


from scipy.spatial import cKDTree
def find_good_neighbors(bad_pixels_mask, good_pixels_mask, topk=10):
    bad_pixels_mask = bad_pixels_mask.cpu().numpy()
    good_pixels_mask = good_pixels_mask.cpu().numpy()
    good_pixels = np.argwhere(good_pixels_mask)
    bad_pixels = np.argwhere(bad_pixels_mask)
    tree = cKDTree(good_pixels)
    dist, idx = tree.query(bad_pixels, k=topk)
    neighbor_good_pixels = good_pixels[idx]  # [n_bad_pixels, topk, 2]
    return bad_pixels, neighbor_good_pixels  # [n_bad_pixels, 2], [n_bad_pixels, topk, 2]


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # test_get_important_uv()
    # test_sample_ggx_vndf()
    # test_sample_imp_light_dir()
    main()

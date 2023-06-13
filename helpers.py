import torch
import os
import numpy as np
import time
import sys
import torch.nn.functional as F
import hdrio
import glob


class LightWrapper(torch.nn.Module):
    def __init__(self, emap_gt, emap_height, n_samples, emap_init='zeros', log_space=False, render_gt=False):
        super(LightWrapper, self).__init__()
        if emap_init == 'zeros':
            if log_space:
                emap_learned = torch.ones(1, 3, emap_height, emap_height * 2) * (-0.5)
                print('Emap initialized as 0.1.')
            else:
                emap_learned = torch.zeros(1, 3, emap_height, emap_height * 2)
                print('Emap initialized as zeros.')
        elif emap_init == 'gt_mean':
            emap_learned = torch.zeros(1, 3, emap_height, emap_height * 2) + emap_gt.mean(dim=[-1, -2], keepdim=True)
            print('Emap initialized as GT mean: {}'.format(emap_learned[0, :, 0, 0].cpu().numpy()))
        else:
            raise NotImplementedError
        self.emap_learned = torch.nn.Parameter(emap_learned)
        self.register_buffer('emap_gt', emap_gt)
        self.n_samples = n_samples
        self.log_space = log_space
        self.render_gt = render_gt

    def forward(self, view_dir, normal_batch, albedo_batch, roughness_batch, metalness_batch):
        imp_light_dir = sample_imp_light_dir(view_dir, roughness_batch, normal_batch, self.n_samples)
        imp_light_uv = map_world_to_latlong_uv(imp_light_dir)  # [P, S, 2]

        grid = imp_light_uv[None, ...]  # [1, P, S, 2]
        if self.render_gt:
            in_radiance = torch.nn.functional.grid_sample(self.emap_gt, grid * 2 - 1, padding_mode='border',
                                                          align_corners=False)
        else:
            emap_learned = 10 ** self.emap_learned if self.log_space else self.emap_learned
            in_radiance = torch.nn.functional.grid_sample(emap_learned, grid * 2 - 1, padding_mode='border',
                                                          align_corners=False)  # [1, 3, P, S]
        in_radiance = in_radiance.squeeze(0).permute(1, 2, 0)
        shaded_batch_spec = shade_principled_bsdf_metal_vndf(albedo_batch, roughness_batch, metalness_batch, normal_batch,
                                                        in_radiance, view_dir, imp_light_dir)  # [P, 3]

        imp_light_dir = sample_imp_light_dir_cos(normal_batch, self.n_samples)
        imp_light_uv = map_world_to_latlong_uv(imp_light_dir)  # [P, S, 2]
        grid = imp_light_uv[None, ...]  # [1, P, S, 2]
        if self.render_gt:
            in_radiance = torch.nn.functional.grid_sample(self.emap_gt, grid * 2 - 1, padding_mode='border',
                                                          align_corners=False)
        else:
            emap_learned = 10 ** self.emap_learned if self.log_space else self.emap_learned
            in_radiance = torch.nn.functional.grid_sample(emap_learned, grid * 2 - 1, padding_mode='border',
                                                          align_corners=False)  # [1, 3, P, S]
        in_radiance = in_radiance.squeeze(0).permute(1, 2, 0)
        shaded_batch_diff = shade_principled_bsdf_diff_vndf(albedo_batch, roughness_batch, metalness_batch, normal_batch,
                                                        in_radiance, view_dir, imp_light_dir)
        shaded_batch = shaded_batch_spec + shaded_batch_diff
        """In the above we use both spec and diff shading, but actually only spec is effective.
        This does not change the estimated lighting because diff is always 0 (metalness is always 1), 
        but it does affect the sampling process and thus the confidence map computation. 
        This is not too wrong if we see this confidence map as a smoothed version, 
        but the follow code snippet is the correct way to do it.
        We used the above code for our paper submission, thus we keep both here."""
        # imp_light_dir = sample_imp_light_dir_cos(normal_batch, self.n_samples)
        # imp_light_uv = map_world_to_latlong_uv(imp_light_dir)  # [P, S, 2]
        # grid = imp_light_uv[None, ...]  # [1, P, S, 2]
        # if self.render_gt:
        #     in_radiance = torch.nn.functional.grid_sample(self.emap_gt, grid * 2 - 1, padding_mode='border',
        #                                                   align_corners=False)
        # else:
        #     emap_learned = 10 ** self.emap_learned if self.log_space else self.emap_learned
        #     in_radiance = torch.nn.functional.grid_sample(emap_learned, grid * 2 - 1, padding_mode='border',
        #                                                   align_corners=False)  # [1, 3, P, S]
        # in_radiance = in_radiance.squeeze(0).permute(1, 2, 0)
        # shaded_batch_diff = shade_principled_bsdf_diff_vndf(albedo_batch, roughness_batch, metalness_batch, normal_batch,
        #                                                 in_radiance, view_dir, imp_light_dir)
        # shaded_batch = shaded_batch_spec
        return shaded_batch, grid

    def clamp_emap(self):
        with torch.no_grad():
            if self.log_space:
                pass  # no need to clamp
            else:
                self.emap_learned.data.clamp_(0)
            return

    def sample_light_grad(self, n_points):
        # sample uv
        emap_learned = 10 ** self.emap_learned if self.log_space else self.emap_learned
        grid = torch.rand(1, 1, n_points, 2) * 2 - 1
        sampled_light = F.grid_sample(emap_learned, grid, padding_mode='border', align_corners=True)  # [1, C, 1, P]
        grid_near = grid + torch.randn_like(grid) * 0.01
        sampled_light_near = F.grid_sample(emap_learned, grid_near, padding_mode='border', align_corners=True)
        return sampled_light, sampled_light_near  # [1, C, 1, P]


class Logger(object):
    def __init__(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        def time_string():
            ISOTIMEFORMAT = '%Y-%m-%d %X'
            string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
            return string

        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_log(self, string):
        self.file.write("{}\n".format(string))
        self.file.flush()
        print(string)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_all_output_dirs(output_dirs, current_depth, target_depth):
    """
    :param output_dirs: list of candidate dirs
    :param current_depth:
    :param target_depth:
    :return:
        a list of all dirs of target_depth
    """
    if not isinstance(output_dirs, list):
        output_dirs = [output_dirs]
    if current_depth == target_depth:
        return output_dirs
    else:
        new_output_dirs = []
        for parent_dir in output_dirs:
            possible_dirs = sorted(glob.glob(os.path.join(parent_dir, '*')))
            child_dirs = list(filter(lambda possible_dir: os.path.isdir(possible_dir), possible_dirs))
            new_output_dirs += child_dirs
        return find_all_output_dirs(new_output_dirs, current_depth+1, target_depth)


def map_world_to_latlong_uv(dir_in_world):
    """
    Get the (u, v) coordinates of the point defined by (x, y, z) for a latitude-longitude map.
    :param dir_in_world: [..., 3]
    :return:
        uv_coord: [..., 2], ranging [0, 1], left-top is (0,0) and right-bottom is (1,1)
    """
    x, y, z = dir_in_world[..., 0], dir_in_world[..., 1], dir_in_world[..., 2]
    u = 1 + (1 / torch.pi) * torch.arctan2(x, -z)
    v = (1 / torch.pi) * torch.arccos(y.clamp(-1, 1))  # to avoid NAN
    # because we want [0,1] interval
    u = u / 2
    uv_coord = torch.stack([u, v], dim=-1)
    return uv_coord


def shade_principled_bsdf_diff_vndf(base_color_map, roughness, metalness, normal_map, in_radiance, view_dir, light_dir):
    """
    :param base_color_map: [P, 3]
    :param roughness: [P, 1]
    :param metalness: [P, 1]
    :param normal_map: [P, 3]
    :param in_radiance: [P, S, 3]
    :param view_dir: [P, 3]
    :param light_dir: [P, S, 3]
    :return:
        shaded_px: [P, 3]
    """
    base_color_map = base_color_map * (1 - metalness)
    shaded_samples = in_radiance * base_color_map[:, None]  # cosine/pi has been canceled by importance sampling
    shaded_px = shaded_samples.mean(dim=1)
    return shaded_px


def shade_principled_bsdf_metal_vndf(base_color_map, roughness, metalness, normal_map, in_radiance, view_dir, light_dir):
    """
    :param base_color_map: [P, 3]
    :param roughness: [P, 1]
    :param metalness: [P, 1]
    :param normal_map: [P, 3]
    :param in_radiance: [P, S, 3]
    :param view_dir: [P, 3]
    :param light_dir: [P, S, 3]
    :return:
        shaded_px: [P, 3]
    """
    half_dir = view_dir[:, None, :] + light_dir  # [P, S, 3]
    half_dir = F.normalize(half_dir, dim=-1)
    normal = normal_map[:, None, :]  # [P, 1, 3]

    hdv = torch.clamp((half_dir * view_dir[:, None, :]).sum(dim=-1), 0, 1)  # [P, S]
    ndv = torch.clamp((normal * view_dir[:, None, :]).sum(dim=-1), 0, 1)  # [P, S]
    ndl = torch.clamp((normal * light_dir).sum(dim=-1), 0, 1)  # [P, S]

    fresnel = compute_fresnel_metal(base_color_map, metalness, hdv)  # [P, S, 3]
    geometric = compute_geometric(roughness, ndv, ndl)  # [P, S]
    smith_view = compute_smith_geometric(alpha=roughness**2, cos=ndv).clamp(1e-10, 1)  # [P, S]

    shaded_samples = fresnel * geometric[..., None] * in_radiance / smith_view[..., None]
    shaded_px = shaded_samples.mean(dim=1)
    return shaded_px


def sample_imp_light_dir_cos(normal, n_samples):
    """ sampling light directions for diffuse surfaces (Lambert BRDF)
    :param normal: [P, 3], in world frame
    :param n_samples:
    :return:
        imp_light_dir: [P, S, 3]
    """
    P = normal.shape[0]

    # local frame
    u, v = torch.rand(P, n_samples, device=normal.device), torch.rand(P, n_samples, device=normal.device)  # [P, S]
    phi = 2.0 * torch.pi * u
    costheta = torch.sqrt(v)
    sintheta = torch.sqrt(1.0 - v)

    # Cartesian vector in local space
    x = torch.cos(phi) * sintheta
    y = torch.sin(phi) * sintheta
    z = costheta
    imp_light_dir_local = torch.stack([x, y, z], dim=-1)  # [P, S, 3]

    tbn_matrix = construct_tbn_matrix(normal)  # [P, 3, 3], world2local rotation
    imp_light_dir = torch.matmul(tbn_matrix.transpose(1, 2)[:, None, ...], imp_light_dir_local[..., None]).squeeze(-1)
    return imp_light_dir


def sample_imp_light_dir(view_dir, rough, normal, n_samples):
    """ Principled bsdf with VNDF (GGX+Smith) sampling for a batch of pixels
    :param view_dir: [P, 3]
    :param rough: [P, 1]
    :param normal: [P, 3], in world frame
    :param n_samples:
    :return:
        imp_light_dir: [P, S, 3]
    """
    # find important samples
    P = normal.shape[0]
    alpha = rough ** 2
    tbn_matrix = construct_tbn_matrix(normal)  # [P, 3, 3], transforms world to local frame
    view_dir_local = torch.matmul(tbn_matrix, view_dir[..., None]).squeeze(-1)  # [P, 3]
    u1, u2 = torch.rand([P, n_samples]), torch.rand([P, n_samples])
    imp_micro_normals_local = sample_ggx_vndf(view_dir_local, alpha, u1, u2)  # [P, S, 3]
    imp_micro_normals = torch.matmul(tbn_matrix.transpose(1, 2)[:, None, ...], imp_micro_normals_local[..., None]).squeeze(-1)
    imp_light_dir = reflect(view_dir[:, None, :], imp_micro_normals)  # [P, S, 3]
    return imp_light_dir


def test_sample_imp_light_dir():
    """
    To run this test, you need to add u1,u2 as arguments to sample_imp_light_dir,
    and comment out their generators. Then add imp_micro_normals_local and imp_micro_normals as return.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    view_dir = torch.tensor([0, 0., 1])[None, :]  # [1, 3]
    n_samples = 1000
    view_dir = F.normalize(view_dir, dim=-1)
    rough = 1
    normal = torch.tensor([-0.01, 0., 1])[None, :]
    normal = F.normalize(normal, dim=-1)

    N = 16 * 16
    u1 = torch.arange(N) / N + 1 / N * 0.5
    u2 = torch.arange(N) / N + 1 / N * 0.5
    u1_, u2_ = torch.meshgrid([u1, u2], indexing='xy')

    idx_u1, idx_u2 = torch.arange(N), torch.arange(N)
    idx_u1, idx_u2 = torch.meshgrid([idx_u1, idx_u2], indexing='xy')  # [N, N]
    res_u1, res_u2 = idx_u1 // 16, idx_u2 // 16
    res = res_u1 + res_u2
    mask = res % 2 == 0
    black = torch.tensor([0.2, 0.2, 0.2])
    white = torch.tensor([0.8, 0.8, 0.8])
    color = torch.ones_like(u1_)[..., None].expand([-1, -1, 3]) * 0.2
    color[mask] = white
    color[~mask] = black

    u1, u2 = u1_, u2_  # [NxN,]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    light_dir, imp_micro_normals_local, imp_micro_normals = sample_imp_light_dir(view_dir, rough, normal, n_samples, u1, u2)

    x_ = light_dir[..., 0] / light_dir[..., 2]
    y_ = light_dir[..., 1] / light_dir[..., 2]
    ax1.axis([-3, 3, -3, 3])
    ax1.scatter(x_.flatten(), y_.flatten(), c=color.flatten(0, 1), s=0.5)
    x_ = imp_micro_normals_local[..., 0] / imp_micro_normals_local[..., 2]
    y_ = imp_micro_normals_local[..., 1] / imp_micro_normals_local[..., 2]
    ax2.axis([-3, 3, -3, 3])
    ax2.scatter(x_.flatten(), y_.flatten(), c=color.flatten(0, 1), s=0.5)
    x_ = imp_micro_normals[..., 0] / imp_micro_normals[..., 2]
    y_ = imp_micro_normals[..., 1] / imp_micro_normals[..., 2]
    ax3.axis([-3, 3, -3, 3])
    ax3.scatter(x_.flatten(), y_.flatten(), c=color.flatten(0, 1), s=0.5)
    plt.show()


def construct_tbn_matrix(normal):
    """
    :param normal: [..., 3], unit norm
    :return:
        tbn_matrix: [..., 3, 3], transforms world coordinates to local frame coordinates
    """
    tmp = torch.tensor([0., 1., 0.])[None, :]
    zero_cross_mask = normal[..., 0] ** 2 + normal[..., -1] ** 2 <= 0  # [...,]
    tangent = F.normalize(torch.linalg.cross(tmp, normal), dim=-1)  # [..., 3]
    tangent[zero_cross_mask] = torch.tensor([1., 0., 0.])
    bitangent = F.normalize(torch.linalg.cross(normal, tangent), dim=-1)
    tbn_matrix = torch.stack([tangent, bitangent, normal], dim=-1).transpose(-2, -1)
    return tbn_matrix


def sample_ggx_vndf(view_dir_ellip, alpha, u1, u2):
    """
    :param view_dir_ellip: [P, 3], in local shading frame (i.e., assuming normal=[0,0,1])
    :param alpha: [P, 1]
    :param u1: [P, S], uniform ranging [0, 1]
    :param u2: [P, S]
    :return:
        micro_normals: [P, S, 3], in local shading frame
    """
    local_normal = torch.tensor([0., 0., 1.])[None, :]  # [1, 3]
    alpha_scale = torch.cat([alpha, alpha, torch.ones_like(alpha)], dim=-1)  # ext [P, 3]
    view_dir_hemi = view_dir_ellip * alpha_scale  # [P, 3]
    view_dir_hemi = F.normalize(view_dir_hemi, dim=-1)

    # find basis
    xy_norm = view_dir_hemi[:, :2].norm(dim=-1)  # [P,]
    zero_cross_mask = xy_norm <= 0  # [P,]
    basis_1 = F.normalize(torch.linalg.cross(local_normal, view_dir_hemi), dim=-1)
    basis_1[zero_cross_mask] = torch.tensor([1., 0., 0.])  # [P, 3]
    basis_2 = F.normalize(torch.linalg.cross(view_dir_hemi, basis_1), dim=-1)  # [P, 3]

    # parameterize the projected area
    r = torch.sqrt(u1)
    theta = 2 * torch.pi * u2
    t1, t2 = r * torch.cos(theta), r * torch.sin(theta)  # [P, S]
    scale = 0.5 * (1 + view_dir_hemi[:, -1:])  # [P, 1]
    t2 = (1 - scale) * torch.sqrt(1 - t1 ** 2) + scale * t2  # [P, S]

    # reproject to hemisphere
    micro_normals_hemi = t1[..., None] * basis_1[:, None, :] + t2[..., None] * basis_2[:, None, :] \
        + torch.sqrt(torch.clamp(1. - t1 ** 2 - t2 ** 2, 0., 1.))[..., None] * view_dir_hemi[:, None, :]  # [P, S, 3]

    # transform micro-normals back to ellipsoid config
    micro_normals = micro_normals_hemi * alpha_scale[:, None, :]
    micro_normals = F.normalize(micro_normals, dim=-1)
    return micro_normals


def test_sample_ggx_vndf():
    import matplotlib.pyplot as plt
    import numpy as np

    """ the following code reproduces Figure 7 """
    N = 16 * 16
    u1 = torch.arange(N) / N + 1/N*0.5
    u2 = torch.arange(N) / N + 1/N*0.5
    u1_, u2_ = torch.meshgrid([u1, u2], indexing='xy')

    idx_u1, idx_u2 = torch.arange(N), torch.arange(N)
    idx_u1, idx_u2 = torch.meshgrid([idx_u1, idx_u2], indexing='xy')  # [N, N]
    res_u1, res_u2 = idx_u1 // 16, idx_u2 // 16
    res = res_u1 + res_u2
    mask = res % 2 == 0
    black = torch.tensor([0.2, 0.2, 0.2])
    white = torch.tensor([0.8, 0.8, 0.8])
    color = torch.ones_like(u1_)[..., None].expand([-1, -1, 3]) * 0.2
    color[mask] = white
    color[~mask] = black

    u1, u2 = u1_.flatten(), u2_.flatten()  # [NxN,]

    alpha = 1
    view_dir_ellip = torch.tensor([1, 0., 1])[None, :]
    view_dir_ellip = F.normalize(view_dir_ellip, dim=-1)
    micro_normals = sample_ggx_vndf(view_dir_ellip, alpha, u1, u2)  # [1, S, 3]
    micro_normals = micro_normals.squeeze()  # [S, 3]
    x_ = - micro_normals[:, 0] / micro_normals[:, 2]
    y_ = - micro_normals[:, 1] / micro_normals[:, 2]
    plt.axis([-3, 3, -3, 3])
    plt.scatter(x_.flatten(), y_.flatten(), c=color.flatten(0, 1), s=0.5)
    plt.show()

    light_dir = reflect(view_dir_ellip, micro_normals)
    x_ = - light_dir[..., 0] / light_dir[..., 2]
    y_ = - light_dir[..., 1] / light_dir[..., 2]
    plt.axis([-3, 3, -3, 3])
    plt.scatter(x_.flatten(), y_.flatten(), c=color.flatten(0, 1), s=0.5)
    plt.show()


def reflect(incident_dir, normal_dir):
    """
    :param incident_dir: [..., 3]
    :param normal_dir: [..., 3]
    :return:
        reflected_dir: [..., 3], normalized
    """
    reflected_dir = - incident_dir + 2 * (incident_dir * normal_dir).sum(dim=-1, keepdim=True) * normal_dir
    reflected_dir = F.normalize(reflected_dir, dim=-1)
    return reflected_dir


def shade_img_naive_imp_samp(albedo, normal, emap, emap_tensor, light_dir, s_u, s_v):
    """
    naive imp samp: sample a [s_u, s_v] rectangular region of envmap around mirror reflection direction
    :param albedo: [P, 3]
    :param normal: [P, 3]
    :param emap: envmap.EnvironmentMap instance
    :param emap_tensor: [H_e, W_e, 3], lighting data used to shade the image
    :param light_dir: [H_e, W_e, 3]
    :param s_u:
    :param s_v:
    :return:
    """
    H_e, W_e, _ = emap_tensor.shape
    emap_list = emap_tensor.flatten(0, 1)  # [H_e x W_e, 3]
    light_dir_list = light_dir.flatten(0, 1)  # [H_e x W_e, 3]

    # find important samples
    view = torch.tensor([0., 0., 1.])
    incident_xyz = - view[None, :] + 2 * (view[None, :] * normal).sum(dim=-1, keepdim=True) * normal
    incident_u, incident_v = emap.world2image(incident_xyz[..., 0].cpu().numpy(),
                                              incident_xyz[..., 1].cpu().numpy(),
                                              incident_xyz[..., 2].cpu().numpy())
    reflected_dir_uv = torch.stack([torch.tensor(incident_u), torch.tensor(incident_v)],
                                   dim=-1)  # [P, 2]
    important_u, important_v, important_idx = get_important_uv(reflected_dir_uv, H_e, W_e, s_u, s_v)  # [P, l], l = s_u * s_v; long index
    P, l = important_idx.shape
    important_idx_flat = important_idx.flatten()

    emap_list_selected = emap_list[important_idx_flat]  # [Pxl, 3]
    emap_tensor_batch = emap_list_selected.reshape([P, l, 3])

    light_dir_list_selected = light_dir_list[important_idx_flat]  # [P*l, 3]
    light_dir_batch = light_dir_list_selected.reshape([P, l, 3])

    colatitude = (important_v.float() + 0.5) / H_e * torch.pi
    inv_density_batch = (s_u / W_e * 2 * torch.pi) * (s_v / H_e * torch.pi) * torch.sin(colatitude)  # [P, l]

    shaded_batch = render_principled_bsdf_metal(albedo, 1., normal, emap_tensor_batch,
                                                light_dir_batch, inv_density_batch)  # [P, 3]
    return shaded_batch


def get_important_uv(center_uv, H_e, W_e, s_u, s_v):
    """
    :param center_uv: [P, 2], uv in an envmap, ranging [0, 1]
    :param H_e: envmap height
    :param W_e: envmap width
    :param s_u: number of samples for u (corresponding to longtitude W_e); must be even number
    :param s_v: number of samples for v (corresponding to colatitude H_e); must be even number
    :return:
        important_u: [P, L = s_u x s_v], long index
        important_v: [P, L = s_u x s_v], long index, used for computing sin(phi) later
        important_idx: [P, L], the index for the uv, considering a flattened [H_e * W_e] dimension
    """
    center_u, center_v = center_uv[..., 0] * W_e, center_uv[..., 1] * H_e  # scale up
    center_u, center_v = center_u.round().long(), center_v.round().long()  # [H, W]
    imp_u_list = []
    imp_v_list = []
    for i in range(s_u):
        imp_u = center_u + i - s_u//2  # [H, W]
        for j in range(s_v):
            imp_v = center_v + j - s_v//2
            imp_u_list.append(imp_u)
            imp_v_list.append(imp_v)
    important_u = torch.stack(imp_u_list, dim=-1)  # [H, W, L]
    important_u = important_u % W_e
    important_v = torch.stack(imp_v_list, dim=-1)  # [H, W, L]
    important_v = important_v % H_e
    important_idx = important_v * W_e + important_u
    return important_u, important_v, important_idx


def compute_fresnel_metal(base_color_map, metalness, hdv):
    """
    :param base_color_map: [P, 3]
    :param metalness: [P, 1]
    :param hdv: [P, L]
    :return:
        fresnel: [P, L, 3]
    """
    base_color_map = metalness * base_color_map + (1 - metalness) * 0.04  # [P, 3]
    base_color_map = base_color_map[..., None, :]  # [P, 1, 3]
    hdv = hdv[..., None]
    fresnel = base_color_map + (1 - base_color_map) * (1 - hdv)**5
    return fresnel


def compute_normal_distribution(roughness, ndh):
    """ GGX normal distribution
    :param roughness: [1,]
    :param ndh: [P, L]
    :return:
        normal_distribution: [P, L]
    """
    alpha = roughness ** 2
    a2 = alpha ** 2
    ndh2 = ndh ** 2
    denom = torch.pi * (ndh2 * (a2 - 1) + 1) ** 2
    normal_distribution = a2 / denom
    return normal_distribution


def compute_geometric(roughness, ndv, ndl):
    """ GGX geometric term
    :param roughness: [P, 1]
    :param ndv: [P, L]
    :param ndl: [P, L]
    :return:
        shadowing: [P, L]
    """
    # alpha = (roughness / 2 + 0.5) ** 2
    alpha = roughness ** 2
    masking = compute_smith_geometric(alpha, ndv)
    shadowing = compute_smith_geometric(alpha, ndl)
    geometric = masking * shadowing
    return geometric


def compute_smith_geometric(alpha, cos):
    """ isotropic GGX smith term
    :param alpha: [P, 1]
    :param cos: [P, L]
    :return:
        smith_term: [P, L]
    """
    a2 = alpha ** 2
    denom_v = cos + torch.sqrt(a2 + (1 - a2) * cos ** 2)
    smith_term = 2 * cos / denom_v
    return smith_term


def render_principled_bsdf_metal(base_color_map, roughness, normal_map, env_map, light_dir, inv_density):
    """
    :param base_color_map: [P, 3]
    :param roughness: [1,]
    :param normal_map: [P, 3]
    :param env_map: [P, L, 3]
    :param light_dir: [P, L, 3], last dim -> xyz
    :param inv_density: extendable to [P, L]
    :return:
    """
    view_dir = torch.tensor([0., 0., 1.])[None, None, :]  # [1, 1, 3]
    half_dir = view_dir + light_dir  # [P, L, 3]
    half_dir = F.normalize(half_dir, dim=-1)  # [P, L, 3]
    normal = normal_map[..., None, :]  # [P, 1, 3]

    hdv = torch.clamp((half_dir * view_dir).sum(dim=-1), 0, 1)  # [P, L]
    ndh = torch.clamp((normal * half_dir).sum(dim=-1), 0, 1)  # [P, L]
    ndv = torch.clamp((normal * view_dir).sum(dim=-1), 0, 1)  # [P, L]
    ndl = torch.clamp((normal * light_dir).sum(dim=-1), 0, 1)  # [P, L]
    # brdf
    fresnel = compute_fresnel_metal(base_color_map, hdv)  # [P, L, 3]
    normal_distribution = compute_normal_distribution(roughness, ndh)  # [P, L]
    geometric = compute_geometric(roughness, ndv, ndl)  # [P, L]
    brdf_cos = fresnel * normal_distribution[..., None] * geometric[..., None] / (4 * ndv[..., None])  # [P, L, 3]

    # integration
    shaded_img = (brdf_cos * env_map * inv_density[..., None]).mean(axis=1)
    return shaded_img


def test_get_important_uv():
    H_e, W_e, s_u, s_v = 1024, 2048, 16, 16
    emap = torch.zeros([H_e, W_e, 3])
    center_uv = torch.tensor([2047, 1000.])
    important_u, important_v, important_idx = get_important_uv(center_uv, H_e, W_e, s_u, s_v)
    emap_flat = emap.flatten(0, 1)
    emap_flat[important_idx] = 1.
    emap_back = emap_flat.reshape([H_e, W_e, 3])
    hdrio.imwrite(emap_back.cpu().numpy(), 'emap_hot.png')

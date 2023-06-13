# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer
import render.optixutils as ou

###############################################################################
#  Geometry interface
###############################################################################

class DLMesh:
    def __init__(self, initial_guess, FLAGS):
        self.FLAGS = FLAGS

        self.initial_guess     = initial_guess
        self.mesh              = initial_guess.clone()

        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()

        if self.FLAGS.opt_cam_pos:
            # cam pos optimization
            # self.rot = torch.eye(3).cuda() # for rotation matrix
            self.init_rot = torch.tensor(FLAGS.rot_init).float().cuda()
            self.init_rot.requires_grad_(False)
            self.rot = torch.tensor(FLAGS.rot_init).float().cuda()
            self.trans = torch.zeros(3, 1).cuda()

            self.rot.requires_grad_(True)
            self.trans.requires_grad_(True)
            
            self.scale = torch.ones((1,)).cuda()
            self.scale.requires_grad_(True)

            print("Initalizing pose...")
            print(self.rot, self.trans, self.scale)

        else:
            self.mesh.v_pos.requires_grad_(True)

        print("Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        print("Avg edge length: %f" % regularizer.avg_edge_length(self.mesh.v_pos, self.mesh.t_pos_idx))

    def parameters(self):
        if self.FLAGS.run_proposed:
            return [self.rot, self.trans]
        elif self.FLAGS.opt_cam_pos:
            return [self.rot, self.trans, self.scale]
        return [self.mesh.v_pos]

    def getOptimizer(self, lr_pos):
        return torch.optim.Adam(self.parameters(), lr=lr_pos)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material):
        self.mesh.material = material

        imesh = mesh.Mesh(base=self.mesh)
        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, imesh.v_pos.contiguous(), imesh.t_pos_idx.int(), rebuild=1)

        if self.FLAGS.opt_cam_pos:
            rot_norm = self.rot/torch.norm(self.rot) # normalizing to unit quaternion
            # print(rot_norm, self.trans, self.scale)
            imesh.v_pos = self.initial_guess.v_pos.clone()
            imesh.move_mesh(rot_norm, self.trans, self.scale)

        # Compute normals and tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        return imesh

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, FLAGS, denoiser):
        
        color_ref = target['img']

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================

        if FLAGS.camera_space_light is None:
            transform = None
        elif FLAGS.convert_frame:
            transform = torch.matmul(target['convert_frame'], target['mv'])
        else:
            transform = target['mv']

        opt_mesh = self.getMesh(opt_material)
        buffers = render.render_mesh(FLAGS, glctx, opt_mesh, target['mvp'], target['campos'], target['light'] if lgt is None else lgt, target['resolution'],
                                       spp=target['spp'], num_layers=FLAGS.layers, msaa=True, background=target['background'], 
                                       optix_ctx=self.optix_ctx, denoiser=denoiser, transform=transform)

        t_iter = iteration / FLAGS.iter

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        # Image-space loss, split into a coverage component and a color component
        def compute_center_loss(mask_1, mask_2):
            """
            args:
                mask_1: [H, W] tensor, each entry value ranging [0, 1]
                mask_2: [H, W] tensor, each entry value ranging [0, 1]
            return:
                L2 loss between the "center of mass" for both masks.
            """
            H, W = mask_1.shape
            Hs, Ws = torch.meshgrid(torch.arange(H) / (H - 1), torch.arange(W) / (W - 1))
            H_mean_1 = (Hs.cuda() * mask_1).sum() / (mask_1.sum() + 1e-5)
            H_mean_2 = (Hs.cuda() * mask_2).sum() / (mask_2.sum() + 1e-5)
            W_mean_1 = (Ws.cuda() * mask_1).sum() / (mask_1.sum() + 1e-5)
            W_mean_2 = (Ws.cuda() * mask_2).sum() / (mask_2.sum() + 1e-5)
            center_loss = (H_mean_1 - H_mean_2) ** 2 + (W_mean_1 - W_mean_2) ** 2
            return center_loss

        color_ref = target['img']
        chamfer_map = target['chamfer_map'][..., 0]  # [1, H, W]
        rendered_mask = buffers['shaded'][..., 3]
        ref_mask = color_ref[..., 3]
        img_loss = (rendered_mask - ref_mask).pow(2).mean()
        img_loss += compute_center_loss(rendered_mask[0], ref_mask[0]) * 100
        img_loss += (chamfer_map * rendered_mask).pow(2).mean()
        img_loss += ((1-rendered_mask) * ref_mask).pow(2).mean()

        img_loss += loss_fn(buffers['shaded'][..., 0:3] * ref_mask[..., None],
                            color_ref[..., 0:3] * ref_mask[..., None])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # rotation regularization
        if t_iter < 0.2:
            reg_loss += torch.norm(self.rot - self.init_rot)

        # Monochrome shading regularizer
        if not self.FLAGS.only_camera_pose and not self.FLAGS.run_proposed:
            reg_loss += regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, FLAGS.lambda_diffuse, FLAGS.lambda_specular)

        # Material smoothness regularizer
        if not self.FLAGS.run_proposed:
            reg_loss += regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.FLAGS.lambda_kd, lambda_ks=self.FLAGS.lambda_ks, lambda_nrm=self.FLAGS.lambda_nrm)

        # Chroma regularizer
        if not self.FLAGS.run_proposed:
            reg_loss += regularizer.chroma_loss(buffers['kd'], color_ref, self.FLAGS.lambda_chroma)

        # Perturbed normal regularizer
        if not self.FLAGS.run_proposed:
            if 'perturbed_nrm_grad' in buffers:
                reg_loss += torch.mean(buffers['perturbed_nrm_grad']) * FLAGS.lambda_nrm2

        # Laplacian regularizer. 
        if not self.FLAGS.run_proposed:
            if self.FLAGS.laplace == "absolute":
                reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * FLAGS.laplace_scale * (1 - t_iter)
            elif self.FLAGS.laplace == "relative":
                reg_loss += regularizer.laplace_regularizer_const(self.mesh.v_pos - self.initial_guess.v_pos, self.mesh.t_pos_idx) * FLAGS.laplace_scale * (1 - t_iter) 

        if self.FLAGS.smooth_envmap and not self.FLAGS.only_camera_pose:
            reg_loss += regularizer.env_smoothness_grad(lgt.base) * self.FLAGS.smooth_envmap_wt

        return img_loss, reg_loss
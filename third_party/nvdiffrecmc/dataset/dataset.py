# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

""" 
Basic dataset interface. 
"""
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, FLAGS=None): 
        super().__init__()
        self.FLAGS = FLAGS

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def getMesh(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        batch_collated = {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'chamfer_map': torch.cat(list([item['chamfer_map'] for item in batch]), dim=0),
        }

        if self.FLAGS is not None and self.FLAGS.convert_frame:
            batch_collated['convert_frame'] = torch.stack(list([item['convert_frame'] for item in batch]))

        return batch_collated
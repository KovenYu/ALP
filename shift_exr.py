import hdrio
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--input_filename', default=None, help='should be ended with .exr')
parser.add_argument('--back_shift', action='store_true')
args = parser.parse_args()
if args.input_filename is None:
    if args.back_shift:
        args.input_filename = 'emap_learned_shifted.exr'
    else:
        args.input_filename = 'emap_learned.exr'

exr_img_path = os.path.join(args.input_dir, args.input_filename)
exr_img = hdrio.imread(exr_img_path)
H, W, _ = exr_img.shape
if args.back_shift:
    shifted = np.roll(exr_img, W//4*1, axis=1)
    hdrio.imwrite(shifted, exr_img_path.replace('_shifted.exr', '.exr'))
else:
    shifted = np.roll(exr_img, -W//4*1, axis=1)
    hdrio.imwrite(shifted, exr_img_path.replace('.exr', '_shifted.exr'))

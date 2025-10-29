import argparse
import numpy as np
import os
import sys
import torch


CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join('..', os.path.dirname(CUR)))

from model import PointPillarsCore

CLASSES = {
    'jetski': 0,
    'smallboat': 1,
    'mediumboat': 2,
    'c-marker': 3,
    'yacht': 4,
    'pole': 5,
    'dinghyboat': 6
}

POINT_CLOUD_RANGE = [4.0, -72., -10., 180., 72., 30.]
z_span = POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]
VOXEL_SIZE = [0.25, 0.25, z_span]

def main(args):
    if not args.no_cuda:
        model = PointPillarsCore(nclasses=len(CLASSES), voxel_size=VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE, prefix='avikus').cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillarsCore(nclasses=len(CLASSES), voxel_size=VOXEL_SIZE, point_cloud_range=POINT_CLOUD_RANGE, prefix='avikus')
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    model.eval()


    print('start to transform pytorch model to onnx')
    max_pillars = 40000
    pillars = torch.randn(max_pillars, 32, 4)
    coors_batch = torch.randint(0, 216, (max_pillars, 3))
    npoints_per_pillar = torch.randint(0, 32, (max_pillars, ))
    npoints_per_pillar = npoints_per_pillar.to(torch.int32)
    if not args.no_cuda:
        pillars = pillars.cuda()
        coors_batch = coors_batch.cuda()
        npoints_per_pillar = npoints_per_pillar.cuda()

    torch.onnx.export(model, (pillars, coors_batch, npoints_per_pillar), args.saved_onnx_path, 
                      export_params=True, opset_version=11, do_constant_folding=True, 
                      input_names=['input_pillars', 'input_coors_batch', 'input_npoints_per_pillar'],
                      dynamic_axes={'input_pillars': {0: 'pillar_num'}, 
                                    'input_coors_batch': {0: 'pillar_num'}, 
                                    'input_npoints_per_pillar': {0: 'pillar_num'}},
                      output_names=['output_x'])
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='../pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--saved_onnx_path', default='../pretrained/model.onnx',
                        help='your saved onnx path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)

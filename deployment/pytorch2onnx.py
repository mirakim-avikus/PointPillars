import argparse
import numpy as np
import os
import sys
import torch


CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR))

from model import PointPillarsCore


def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'motorboat': 2
        }

    prefix = args.prefix
    if prefix == 'avikus':
        point_cloud_range=[0, -50., -10., 250., 50., 10.]
    else:
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]
    voxel_size=[0.16, 0.16, 4]

    if not args.no_cuda:
        if prefix == 'avikus':
            model = PointPillarsCore(nclasses=len(CLASSES), voxel_size=voxel_size, point_cloud_range=point_cloud_range, prefix='avikus').cuda()
        else:
            model = PointPillarsCore(nclasses=len(CLASSES)).cuda()
        
        model.load_state_dict(torch.load(args.ckpt))
    else:
        if prefix == 'avikus':
            model = PointPillarsCore(nclasses=len(CLASSES), voxel_size=voxel_size, point_cloud_range=point_cloud_range, prefix='avikus')
        else:
            model = PointPillarsCore(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    model.eval()


    print('start to transform pytorch model to onnx')
    max_pillars = 40000
    pillars = torch.randn(max_pillars, 32, 4)
    coors_batch = torch.randint(0, 216, (max_pillars, 4))
    coors_batch[:, 0] = 0
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
    parser.add_argument('--prefix', required=True,
                        help='choose either avikus or kitti')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)

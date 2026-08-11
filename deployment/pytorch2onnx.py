import argparse
import numpy as np
import torch

from pointpillars.dataset import Avikus, POINT_CLOUD_RANGE
from pointpillars.model import PointPillarsCore

CLASSES = Avikus.CLASSES

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

    # nms_pre caps how many anchors survive the in-graph TopK. With the
    # default (100), near-duplicate anchors of the few highest-scoring objects
    # consume every slot, so dense maritime frames (25+ targets) emit only
    # 2-5 boxes no matter the score threshold. Raise it so each object can
    # reach NMS.
    model.nms_pre = args.nms_pre
    print(f'nms_pre = {model.nms_pre}')

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
    print(f'finished, saved to {args.saved_onnx_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='../best.pth',
                        help='checkpoint to export (../best.pth is the 10-class '
                             'avikus model matching the deployed onnx)')
    parser.add_argument('--saved_onnx_path', default='../pretrained/model_nmspre1000.onnx',
                        help='your saved onnx path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--nms_pre', type=int, default=1000,
                        help='anchors kept by the in-graph TopK before NMS '
                             '(100 = original; too small for dense scenes)')
    args = parser.parse_args()

    main(args)

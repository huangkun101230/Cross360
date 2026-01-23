import os
import argparse

import torch
import torch.nn.functional as F

import numpy as np

import datasets
from torch.utils.data import DataLoader

from Networks.network import OursNetwork

import tqdm
from Metrics.depth_metrics import Depth_Evaluator
import utils.file_utils as fu


def main():
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--mask_ratio', type=float, default=0.25, help='valid pixel ratio')
    

    args = parser.parse_args()
    use_cuda = True
    print("use_cuda:", use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    np.random.seed(args.seed)


    val_dataset = datasets.Dataset3D60(val_file_list, 256, 512, is_training=False)
    # val_dataset = datasets.Structured3d(val_file_list, 256, 512, is_training=False)
    val_loader = DataLoader(val_dataset, 1, shuffle=False,
                                 num_workers=4, pin_memory=True, drop_last=True)

    model = OursNetwork()


    if torch.cuda.is_available():
        model.cuda()
                                                
    checkpoint = torch.load(model_path,map_location='cuda:0')

    """loading my model"""
    pretrained_dict = checkpoint['model']
    model.load_state_dict(pretrained_dict)    

    start_epoch = checkpoint['epoch']
    print('loading epoch {} successfully！'.format(start_epoch))

    pbar = tqdm.tqdm(val_loader)
    pbar.set_description("Evaluating")

    depth_evaluator = Depth_Evaluator()
    depth_evaluator.reset_eval_metrics()

    with torch.no_grad():
        for batch_idx, val_sample in enumerate(pbar):

            ori_rgb = val_sample["ori_rgb"].to(device)
            depth = val_sample["gt_depth"].to(device)
            mask = val_sample["depth_mask"].to(device)
            mask = mask>0

            outputs = model(ori_rgb)

            pred_depth = outputs["pred_depth"] * mask
            pred_depth = torch.median(depth) / torch.median(pred_depth) * pred_depth
            depth_evaluator.compute_eval_metrics(depth[mask].detach().cpu(), pred_depth[mask].detach().cpu())

            fu.save_depth_tensor_as_float(output_path, val_sample['depth_filename'][0]+'_pred', pred_depth[0])
            fu.save_depth_tensor_as_float(output_path, val_sample['depth_filename'][0]+'_gt', depth[0])

        depth_evaluator.print(eva_path)

if __name__ == "__main__":
    folder = 'saved_models'

    val_file_list = "datasets/3d60_examples.txt"
    # val_file_list = "datasets/3d60_test.txt"
    # val_file_list = "datasets/Matterport3D_test.txt"
    # val_file_list = "datasets/stanford2d3d_test.txt"
    # val_file_list = "datasets/SunCG_test.txt"
    # val_file_list = "datasets/structured3d_test.txt"

    model_path = './'+folder+'/models/3d60_model.pkl'
    print(model_path)
    
    output_path = './results/'+folder+'/results/'
    eva_path = './results/'+folder+'/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main()
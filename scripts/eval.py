import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from models.refnet import RefNet
from utils.box_util import get_3d_box, box3d_iou
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview
    )
    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader

def get_model(args):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels
    ).cuda()

    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scanrefer(args):
    scanrefer = SCANREFER
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.num_scenes != -1:
        scene_list = scene_list[:args.num_scenes]

    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def evaluate(args):
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args)

    # config
    POST_DICT = {
        'remove_empty_box': True, 
        'use_3d_nms': True, 
        'nms_iou': 0.25,
        'use_old_type_nms': False, 
        'cls_nms': True, 
        'per_class_proposal': True,
        'conf_thresh': 0.05,
        'dataset_config': DC
    } if not args.no_nms else None

    # evaluate
    print("evaluating...")
    ref_acc = []
    ious = []
    masks = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        data = model(data)
        _, data = get_loss(data, DC, True, True, POST_DICT)

        ref_acc += data["ref_acc"]
        ious += data["ref_iou"]
        masks += data["ref_multiple_mask"]

    # aggregate scores
    ref_acc = np.array(ref_acc)
    ious = np.array(ious)
    masks = np.array(masks)

    stats = {
        "unique": np.sum(masks == 0),
        "multiple": np.sum(masks == 1),
        "overall": masks.shape[0]
    }
    scores = {
        "unique": {},
        "multiple": {},
        "overall": {}
    }
    scores["unique"]["ref_acc"] = np.mean(ref_acc[masks == 0]) if np.sum(masks == 0) > 0 else 0
    scores["unique"]["iou_rate_0.25"] = ious[masks == 0][ious[masks == 0] >= 0.25].shape[0] / ious[masks == 0].shape[0] if np.sum(masks == 0) > 0 else 0
    scores["unique"]["iou_rate_0.5"] = ious[masks == 0][ious[masks == 0] >= 0.5].shape[0] / ious[masks == 0].shape[0] if np.sum(masks == 0) > 0 else 0
    scores["multiple"]["ref_acc"] = np.mean(ref_acc[masks == 1]) if np.sum(masks == 1) > 0 else 0
    scores["multiple"]["iou_rate_0.25"] = ious[masks == 1][ious[masks == 1] >= 0.25].shape[0] / ious[masks == 1].shape[0] if np.sum(masks == 1) > 0 else 0
    scores["multiple"]["iou_rate_0.5"] = ious[masks == 1][ious[masks == 1] >= 0.5].shape[0] / ious[masks == 1].shape[0] if np.sum(masks == 1) > 0 else 0
    scores["overall"]["ref_acc"] = np.mean(ref_acc)
    scores["overall"]["iou_rate_0.25"] = ious[ious >= 0.25].shape[0] / ious.shape[0]
    scores["overall"]["iou_rate_0.5"] = ious[ious >= 0.5].shape[0] / ious.shape[0]

    print("done!")

    return stats, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--num_proposals', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_lang_cls', action='store_true', help='Do NOT use language classifier.')
    parser.add_argument('--no_nms', action='store_true', help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    stats, scores = evaluate(args)

    # report
    print("\nstats:")
    for key in stats.keys():
        print("{}:{}".format(key, stats[key]))

    for key in scores.keys():
        print("\n{}:".format(key))
        for metric in scores[key]:
            print("{}: {}".format(metric, scores[key][metric]))

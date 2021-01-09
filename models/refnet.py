import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
#from models.backbone_module import Pointnet2Backbone
#from models.voting_module import VotingModule
#from models.proposal_module import ProposalModule

# new segemnation pipeline - PointGroup
from model.pointgroup.pointgroup import PointGroup 
from model.pointgroup.pointgroup import model_fn_decorator
from util.config import cfg

from models.lang_module import LangModule
from models.match_module import MatchModule

class RefNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference


        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        #self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        #self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        #self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

        ### replace segmentation pipeline with PointGroup ###
        # --config config/pointgroup_run1_scannet.yaml needs to be done as well when calling the train function 
        # TODO: passed in cfg have to be either forwarded from train.py or statically imported into this doc
        self.pointgroup = PointGroup(cfg)

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size)

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        #data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        #xyz = data_dict["fp2_xyz"]
        #features = data_dict["fp2_features"]
        #data_dict["seed_inds"] = data_dict["fp2_inds"]
        #data_dict["seed_xyz"] = xyz
        #data_dict["seed_features"] = features
        
        #xyz, features = self.vgen(xyz, features)
        #features_norm = torch.norm(features, p=2, dim=1)
        #features = features.div(features_norm.unsqueeze(1))
        #data_dict["vote_xyz"] = xyz
        #data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        #data_dict = self.proposal(xyz, features, data_dict)

        # --------- PointGroup ---------
        # needs of next modules self.lang and self.match
        # self.lang: "lang_feat" are already in data_dict
        # self.match: "aggregated_vote_features", 
        #             "objectness_scores" (not necessary with PG)
        #             (both from self.proposal)
        #             "lang_emb" (comes from self.lang)

        # in PointGroup train.py/train_epoch: (data_dict has to be batch)
        # loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)
        # TODO: make sure correct data_dict is given to this forward
        # data_dict needs to conatin: 
        #       - coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        #       - voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        #       - p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        #       - v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda
        #       - coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        #       - feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        #       - labels = batch['labels'].cuda()                        # (N), long, cuda
        #       - instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        #       - instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        #       - instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        #       - batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        #       - spatial_shape = batch['spatial_shape']
        #       - EXTRA: epoch = batch['epoch']

        # loss, visual_dict, meter_dict not necessary here
        # forwarding to downstream app?
        _, preds, _, _ = model_fn(data_dict, self.pointgroup(), data_dict['epoch'])
        # preds['score_feats'] has to be of dim.: [batch_size, num_proposal, 128]
        # that means in PointGroup.forward: dim. C = 128
        assert(preds['score_feats'].shape[-1] == 128)
        data_dict['aggregated_vote_features'] = preds['score_feats']

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict

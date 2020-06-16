""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from utils.nn_distance import nn_distance

def decode_scores(net, data_dict, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
    net_transposed = net.transpose(2,1).contiguous() # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    data_dict['objectness_scores'] = objectness_scores
    
    base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    data_dict['center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    data_dict['size_scores'] = size_scores
    data_dict['size_residuals_normalized'] = size_residuals_normalized
    data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    data_dict['sem_cls_scores'] = sem_cls_scores
    return data_dict

class RefModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, use_lang_classifier=True, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        # --------- FEATURE FUSION ---------
        self.gru = nn.GRU(
            input_size=300,
            hidden_size=256,
            batch_first=True
        )
        self.lang_sqz = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.feat_fuse = nn.Sequential(
            nn.Conv1d(in_channels=128 + 128, out_channels=128, kernel_size=1),
            nn.ReLU()
        )

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(128, 18),
                nn.Dropout()
            )
            
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = nn.Conv1d(128,128,1)
        self.conv2 = nn.Conv1d(128,128,1)
        self.conv3 = nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.conv4 = nn.Conv1d(128,1,1)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        sample_inds = fps_inds

        data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)

        data_dict = decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        # --------- FEATURE FUSION ---------
        lang_feat = data_dict["lang_feat"]
        lang_feat = pack_padded_sequence(lang_feat, data_dict["lang_len"], batch_first=True, enforce_sorted=False)
    
        # encode description
        _, lang_feat = self.gru(lang_feat)
        data_dict["lang_emb"] = lang_feat
        lang_feat = self.lang_sqz(lang_feat.squeeze(0)).unsqueeze(2).repeat(1, 1, self.num_proposal)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(lang_feat[:, :, 0])
        
        # fuse
        features = self.feat_fuse(torch.cat([features, lang_feat], dim=1))

        # --------- REFERENCE PREDICTION ---------
        masked_features = features * data_dict['objectness_scores'].max(2)[1].float().unsqueeze(1).repeat(1, 128, 1)
        
        data_dict['cluster_ref'] = self.conv4(masked_features).squeeze(1)
        
        return data_dict


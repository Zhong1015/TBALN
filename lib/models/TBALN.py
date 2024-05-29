from collections import OrderedDict
import os
import warnings
import torchvision
import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from .transformer_tdrg.transformer_tdrg import build_transformer
import models


from utils.misc import clean_state_dict, clean_body_state_dict

from .transformer_tdrg.positional_encoding import build_position_encoding
import math
import numpy as np

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x    
    
class TBAL(nn.Module):
    def __init__(self, model, num_classes):
        super(TBAL, self).__init__()
        # backbone
        self.layer1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            # model.layer2,
            # model.layer3,
            # model.layer4,
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        # self.backbone = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        # hyper-parameters
        self.num_classes = num_classes
        self.in_planes = 2048
        self.transformer_dim = 2048
        # self.gcn_dim = 512
        self.num_queries = num_classes
        self.n_head = 4
        self.num_encoder_layers = 2
        self.num_decoder_layers = 3

        # transformer
        self.transform_14 = nn.Conv2d(self.in_planes, self.transformer_dim, 1)
        self.transform_28 = nn.Conv2d(self.in_planes // 2, self.transformer_dim, 1)
        self.transform_7 = nn.Conv2d(self.in_planes, self.transformer_dim, 3, stride=2)

        # Random embedding
        self.query_embed_1 = nn.Embedding(self.num_queries, self.transformer_dim)
        self.query_embed_2 = nn.Embedding(self.num_queries, self.transformer_dim)
        self.query_embed_3 = nn.Embedding(self.num_queries, self.transformer_dim)
        
        self.positional_embedding = build_position_encoding(hidden_dim=self.transformer_dim, mode='learned')
        self.transformer = build_transformer(d_model=self.transformer_dim, nhead=self.n_head,
                                             num_encoder_layers=self.num_encoder_layers,
                                             num_decoder_layers=self.num_decoder_layers)


        self.fc = GroupWiseLinear(num_classes, self.transformer_dim * 3, bias=True)



    def forward_backbone(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

    @staticmethod
    def cross_scale_attention(x3, x4, x5):
        h3, h4, h5 = x3.shape[2], x4.shape[2], x5.shape[2]
        h_max = max(h3, h4, h5)
        x3 = F.interpolate(x3, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h_max, h_max), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h_max, h_max), mode='bilinear', align_corners=True)

        mul = x3 * x4 * x5
        x3 = x3 + mul
        x4 = x4 + mul
        x5 = x5 + mul

        x3 = F.interpolate(x3, size=(h3, h3), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(h4, h4), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(h5, h5), mode='bilinear', align_corners=True)
        return x3, x4, x5

    def forward_transformer(self, x3, x4):
        # cross scale attention
        x5 = self.transform_7(x4)
        x4 = self.transform_14(x4)
        x3 = self.transform_28(x3)
        
        x3, x4, x5 = self.cross_scale_attention(x3, x4, x5)

        # transformer encoder
        mask3 = torch.zeros_like(x3[:, 0, :, :], dtype=torch.bool).cuda()
        mask4 = torch.zeros_like(x4[:, 0, :, :], dtype=torch.bool).cuda()
        mask5 = torch.zeros_like(x5[:, 0, :, :], dtype=torch.bool).cuda()

        pos3 = self.positional_embedding(x3)
        pos4 = self.positional_embedding(x4)
        pos5 = self.positional_embedding(x5)

        feat3 = self.transformer(x3, mask3, self.query_embed_1.weight, pos3)[0]
        feat4 = self.transformer(x4, mask4, self.query_embed_2.weight, pos4)[0]
        feat5 = self.transformer(x5, mask5, self.query_embed_3.weight, pos5)[0] # shape=[21,80,512]
        
        feat = torch.cat((feat3[0], feat4[0], feat5[0]), dim=2)
        
        feat =self.fc(feat)

        
        return feat


    def forward(self, x):
        x2, x3, x4 = self.forward_backbone(x)

        # structural relation
        out = self.forward_transformer(x3, x4)

        return out
    
def TBALN(args):
    
    model = TBAL(
        model =torchvision.models.resnet101(pretrained=True),
        num_classes=args.num_class
    )

    return model
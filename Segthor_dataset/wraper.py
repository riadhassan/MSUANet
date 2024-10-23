import sys
sys.path.append('../')

import torch
import torch.optim as optim
from network_lib.networks import EMCADNet
import numpy as np
import torchvision
from torch.autograd import Variable
from utils import *
from torch.nn.modules.loss import CrossEntropyLoss
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume
import logging
from tensorboardX import SummaryWriter


class ModelWraper:
    def __init__(self, conf):
        self.conf = conf
        self.device = torch.device(conf.device)
        if self.conf.network_type == "EMCAD":
            self.seg_model = EMCADNet(num_classes=conf.num_classes, kernel_sizes=conf.kernel_sizes,
                             expansion_factor=conf.expansion_factor, dw_parallel=not conf.no_dw_parallel,
                             add=not conf.concatenation, lgag_ks=conf.lgag_ks, activation=conf.activation_mscb,
                             encoder=conf.encoder, pretrain=not conf.no_pretrain)
        self.num_classes = conf.num_classes
        self.base_lr = conf.base_lr
        self.seg_model.to(self.device)
        self.weights = None
        self.optimizer1 = optim.AdamW(self.seg_model.parameters(), lr=self.base_lr, weight_decay=0.0001)
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(self.num_classes)
        self.writer = SummaryWriter(conf.snapshot_path + '/log')
        self.supervision = conf.supervision


    def set_mood(self, Train=True):
        if Train:
            self.seg_model.train()
        else:
            self.seg_model.eval()

    def update_models(self, input, iter, epoch):
        image_batch, label_batch = input[0], input[1]
        image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()

        P = self.seg_model(image_batch, mode='train')



        return P


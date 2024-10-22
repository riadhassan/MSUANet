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
        if self.conf.network_type == "Unet":
            self.seg_model = EMCADNet(num_classes=conf.num_classes, kernel_sizes=conf.kernel_sizes,
                             expansion_factor=conf.expansion_factor, dw_parallel=not conf.no_dw_parallel,
                             add=not conf.concatenation, lgag_ks=conf.lgag_ks, activation=conf.activation_mscb,
                             encoder=conf.encoder, pretrain=not conf.no_pretrain)
        self.num_classes = conf.num_classes
        self.base_lr = conf.base_lr
        self.seg_model.to(self.device)
        self.weights = None
        self.optimizer1 = optim.Adam(self.seg_model.parameters(), lr=conf.lr)
        self.optimizer1 = optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=0.0001)
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(self.num_classes)
        writer = SummaryWriter(snapshot_path + '/log')


    def set_mood(self, Train=True):
        if Train:
            self.seg_model.train()
        else:
            self.seg_model.eval()

    def update_models(self, input, iter, epoch):
        image_batch, label_batch = input[0], input[1]

        P = self.seg_model(image_batch, mode='train')

        if not isinstance(P, list):
            P = [P]
        if epoch == 0 and iter == 0:
            n_outs = len(P)
            out_idxs = list(np.arange(n_outs))  # [0, 1, 2, 3]#, 4, 5, 6, 7]
            if conf.supervision == 'mutation':
                ss = [x for x in powerset(out_idxs)]
            elif conf.supervision == 'deep_supervision':
                ss = [[x] for x in out_idxs]
            else:
                ss = [[-1]]
            print(ss)

        loss = 0.0
        w_ce, w_dice = 0.3, 0.7

        for s in ss:
            iout = 0.0
            if (s == []):
                continue
            for idx in range(len(s)):
                iout += P[s[idx]]
            loss_ce = self.ce_loss(iout, label_batch[:].long())
            loss_dice = self.dice_loss(iout, label_batch, softmax=True)
            loss += (w_ce * loss_ce + w_dice * loss_dice)

        self.optimizer1.zero_grad()
        self.loss.backward()
        self.optimizer1.step()
        # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
        lr_ = self.base_lr
        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = lr_

        iter_num = iter + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)

        if iter_num % 50 == 0:
            logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch, loss.item(), lr_))

        return loss

    logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter, epoch, loss.item(), lr_))

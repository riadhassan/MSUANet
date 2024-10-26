import numpy as np
import torch
import argparse
import torch.nn as nn
from Segthor_dataset.data_loader import *
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat
import Segthor_dataset.evaluate as evaluate
import nibabel as nib
import pandas as pd
import os
from Segthor_dataset.wraper import ModelWraper
import random
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume
import logging
from tensorboardX import SummaryWriter
from network_lib.networks import EMCADNet
import torch.optim as optim

def save_validation_nifti(img, gt, seg, path, patient, affine):
    new_img = nib.Nifti1Image(img, affine)
    nib.save(new_img, path + f"/{patient[0]}.nii.gz")
    new_img = nib.Nifti1Image(seg, affine)
    nib.save(new_img, path + f"/{patient[0]}_mask.nii.gz")
    new_img = nib.Nifti1Image(gt, affine)
    nib.save(new_img, path + f"/{patient[0]}_GT.nii.gz")



def conf():
    args = argparse.ArgumentParser()
    args.add_argument("--data_root", type=str, default="C:\\Users\IICT2\Desktop\Dataset_SegThor")
    args.add_argument("--input_channels", type=int, default=1)
    args.add_argument("--output_channels", type=int, default=5)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--save_dir", type=str, default="/output")
    args.add_argument("--model_name", type=str, default="test2")
    args.add_argument("--printfq", type=int, default=50)
    args.add_argument("--writerfq", type=int, default=50)
    args.add_argument("--model_save_fq", type=bool, default=False)
    args.add_argument("--debug_type", type=str, default="nifti", help="Two options: 1) nifti. 2)jpg")
    args.add_argument("--num_epoch", type=int, default=200)
    args.add_argument("--done_epoch", type=int, default=0)
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--imsize", type=int, default=256)
    args.add_argument("--network_type", type=str, default="EMCAD")
    args.add_argument('--dataset', type=str,
                        default='SegThor', help='experiment_name')
    args.add_argument('--list_dir', type=str,
                        default='C:\\Users\IICT2\PycharmProjects\EMCAD\lists\lists_Synapse', help='list dir')
    args.add_argument('--num_classes', type=int,
                        default=5, help='output channel of network')
    # network related parameters
    args.add_argument('--encoder', type=str,
                        default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
    args.add_argument('--expansion_factor', type=int,
                        default=2, help='expansion factor in MSCB block')
    args.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
    args.add_argument('--lgag_ks', type=int,
                        default=3, help='Kernel size in LGAG')
    args.add_argument('--activation_mscb', type=str,
                        default='relu6', help='activation used in MSCB: relu6 or relu')
    args.add_argument('--no_dw_parallel', action='store_true',
                        default=False, help='use this flag to disable depth-wise parallel convolutions')
    args.add_argument('--concatenation', action='store_true',
                        default=False, help='use this flag to concatenate feature maps in MSDC block')
    args.add_argument('--no_pretrain', action='store_true',
                        default=False, help='use this flag to turn off loading pretrained enocder weights')
    args.add_argument('--supervision', type=str,
                        default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

    args.add_argument('--max_iterations', type=int,
                        default=50000, help='maximum epoch number to train')
    args.add_argument('--max_epochs', type=int,
                        default=300, help='maximum epoch number to train')
    args.add_argument('--batch_size', type=int,
                        default=6, help='batch_size per gpu')
    args.add_argument('--base_lr', type=float, default=0.0001,
                        help='segmentation network learning rate')
    args.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    args.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    args.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    args.add_argument('--seed', type=int,
                        default=2222, help='random seed')
    args = args.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'SegThor': {
            'root_path': args.data_root,
            'volume_path': args.data_root,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if args.concatenation:
        aggregation = 'concat'
    else:
        aggregation = 'add'

    if args.no_dw_parallel:
        dw_mode = 'series'
    else:
        dw_mode = 'parallel'

    run = "SegThor_Only_EMCAD_With_Noise_fusion_"
    args.exp = args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(
        args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
        run) + '_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(
        args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
        run))
    snapshot_path = snapshot_path.replace('[', '').replace(']', '').replace(', ', '_')

    snapshot_path = snapshot_path + '_pretrain' if not args.no_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 50000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 300 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.0001 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    args.snapshot_path = snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    return args


def create_dirs(conf):
    if not os.path.exists(os.path.join(conf.save_dir, conf.model_name)):
        os.mkdir(os.path.join(conf.save_dir, conf.model_name))
    model_path = os.path.join(conf.save_dir, conf.model_name)
    if not os.path.exists(os.path.join(conf.save_dir, conf.model_name, "curves")):
        os.mkdir(os.path.join(conf.save_dir, conf.model_name, "curves"))
        os.mkdir(os.path.join(conf.save_dir, conf.model_name, "debug"))

    curve_path = os.path.join(conf.save_dir, conf.model_name, "curves")
    debug_path = os.path.join(conf.save_dir, conf.model_name, "debug")
    return model_path, curve_path, debug_path


def run_on_slices(model, data, conf):
    seg_mask = []
    gt_mask = []
    img_vol = []
    for sl in data:
        mat = loadmat(sl[0])
        if 'affine' in mat.keys():
            affine = mat['affine']
            header = mat['header']

        mask = mat['mask']
        image = mat['img']
        mask = mask[None, :, :]
        # image[image<-150] = -150
        # image[image>200] =200
        # image = image + 150
        # image = (2*image)/200 -1
        image = image / image.max()
        # image = (image - np.std(image))/np.mean(image)
        image = torch.from_numpy(image[None, None, :, :].astype(np.float32))
        image = image.to(conf.device)
        _, pred = model(image)
        pred = nn.Softmax(pred, axis=1)
        pred = torch.argmax(pred, axis=1)
        gt_mask.append(mask)
        seg_mask.append(pred.detach().cpu().numpy())
        img_vol.append(image.detach().cpu().numpy())

    img_vol = np.transpose(np.squeeze(np.asarray(img_vol)).astype(np.float32), (1, 2, 0))
    gt_mask = np.transpose(np.squeeze(np.asarray(gt_mask)).astype(np.uint8), (1, 2, 0))
    seg_mask = np.transpose(np.squeeze(np.asarray(seg_mask)).astype(np.uint8), (1, 2, 0))

    return img_vol, gt_mask, seg_mask, affine


def main(conf):
    # device = torch.device("cpu" if not torch.cuda.is_available() else "mps")
    device = torch.device(conf.device)
    # wraper = ModelWraper(conf)
    seg_model = EMCADNet(num_classes=conf.num_classes, kernel_sizes=conf.kernel_sizes,
                              expansion_factor=conf.expansion_factor, dw_parallel=not conf.no_dw_parallel,
                              add=not conf.concatenation, lgag_ks=conf.lgag_ks, activation=conf.activation_mscb,
                              encoder=conf.encoder, pretrain=not conf.no_pretrain)
    seg_model.to(device)
    optimizer = optim.AdamW(seg_model.parameters(), lr=conf.base_lr, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(conf.num_classes)
    writer = SummaryWriter(conf.snapshot_path + '/log')
    train_loader, val_loader = data_loaders(conf.data_root)

    loaders = {"train": train_loader, "valid": val_loader}
    # model_path, log_path, debug_path = create_dirs(conf)
    conf.debug_path = conf.snapshot_path

    all_dice_dict = []
    all_asd_dict = []
    iter_num = 0

    ###### Training #######
    total_iter = 0
    for epoch in tqdm(range(conf.done_epoch, conf.num_epoch + 1)):
        print("Training...")
        #### Training Loop ###
        seg_model.train()
        conf.curr_epoch = epoch

        for i, data in enumerate(train_loader):
            image_batch, label_batch = data[0], data[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()


            P = seg_model(image_batch, mode='train')


            if not isinstance(P, list):
                P = [P]
            if epoch == 0 and i == 0:
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
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = conf.base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch, loss.item(), lr_))
                print('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch, loss.item(), lr_))

        print(f"End of epoch: {epoch}. Now validating.....")
        seg_model.eval()
        all_dice = []
        all_asd = []
        all_iou = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                vdata, patient = data
                img_vol, gt, seg, affine_mat = run_on_slices(seg_model, vdata, conf)
                dice, asd, iou = evaluate.evaluate_case(seg, gt, evaluate.get_Organ_regions())
                all_dice.append(dice)
                all_asd.append(asd)
                all_iou.append(iou)

        organ_dice = np.mean(all_dice, 0)
        organ_asd = np.mean(all_asd, 0)
        organ_iou = np.mean(all_iou, 0)
        dice_dict, asd_dict = evaluate.print_Thoracic(organ_dice, organ_asd)

        print(dice_dict)
        print(asd_dict)
        all_dice_dict.append(dice_dict)
        all_asd_dict.append(asd_dict)
        pd.DataFrame.from_dict(all_dice_dict).to_csv(os.path.join(conf.snapshot_path, "Validation_dice.csv"))
        pd.DataFrame.from_dict(all_asd_dict).to_csv(os.path.join(conf.snapshot_path, "Validation_asd.csv"))
        # pd.DataFrame.from_dict(all_iou_dict).to_csv(os.path.join(model_path,"Validation_iou.csv"))

        ### Saving the best model ###
        if epoch < 1:
            best_mean_dice = 0


        curr_mean_dice = np.mean(organ_dice)
        print(f"Epoch: {epoch} Mean dice: {curr_mean_dice} Best Dice: {best_mean_dice}")
        if curr_mean_dice > best_mean_dice:
            torch.save({
                'epoch': epoch,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(conf.snapshot_path, "best.pth"))
            best_mean_dice = curr_mean_dice
        elif epoch%50==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(conf.snapshot_path, f"model_{epoch}.pth"))
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': seg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(conf.snapshot_path, "last.pth"))


if __name__ == "__main__":
    main(conf())
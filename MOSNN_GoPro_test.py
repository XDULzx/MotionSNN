import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from models.fusion_models import Fusion_MOSNN
import data_loaders
from functions import TET_loss, seed_all
from functions import CharbonnierLoss, EdgeLoss
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import autocast as autocast
# from torch.cuda.amp import GradScaler
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j',
                    '--workers',
                    default=12,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=24,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-4,  #9.987e-4
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    default=False,
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=600,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=0.0,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--outpath',
                    type=str,
                    default='MOSNN_out',
                    help='output dir path')
args = parser.parse_args()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    # args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    checkpoint_name = None # "20230503_225309"
    if not checkpoint_name:
        checkpoint_name = time.strftime("MOSNN_GoPro_%Y%m%d_%H%M%S", time.localtime())

    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:2456',
                            world_size=args.nprocs,
                            rank=local_rank)
    load_names = None
    save_names = 'gopro_T10_model_distribute.pth'
    load_names = os.path.join(args.outpath, 'MOSNN_GoPro_20230715_172358', 'checkpoints', 'best_35.6560_0.9780_7232_gopro_T10_model_distribute.pth')

    model = Fusion_MOSNN(imgchannel=3, eventchannel=2, outchannel=3)
    model.T = args.T

    if args.local_rank == 0:
        total = sum([param.nelement() for param in model.parameters()])
        print("total params is {:.4f} MB".format(total / 1e6))

    if load_names != None:
        # state_dict = torch.load(load_names)
        model.load_state_dict(torch.load(load_names, map_location='cuda:{}'.format(args.local_rank)), strict=False)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      find_unused_parameters=True)


    # define loss function (criterion) and optimizer
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss().cuda(local_rank)
    criterion_edge.kernel = criterion_edge.kernel.cuda(local_rank)
    # criterion = nn.CrossEntropyLoss().cuda(local_rank)


    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset = data_loaders.build_gopro(spikedatasetPath="/data/LZX/Datasets/gopropbs2_new/data/ptevents_T10/",
                                                          bulrPath="/data/LZX/Datasets/gopropbs2_new/data/blur/",
                                                          gtimgPath="/data/LZX/Datasets/gopropbs2_new/data/gt/",
                                                          train_index_file="/data/LZX/Datasets/gopropbs2_new/list/rand/Trainlist.txt",
                                                          test_index_file="/data/LZX/Datasets/gopropbs2_new/list/rand/Testlist.txt",
                                                          crop=512)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        val_sampler.set_epoch(epoch)

        # evaluate on validation set
        val_start = time.time()
        val_PSNR, val_SSIM = validate(val_loader, model, criterion_edge, criterion_char, local_rank, args)
        val_t = time.time() - val_start

        scheduler.step()

        if local_rank == 0:
            print('Test GoPro: PSNR: {:.4f}, SSIM: {:.4f}'.format(val_PSNR, val_SSIM))
            print('################################### test end #########################################')



def validate(val_loader, model, criterion_edge, criterion_char, local_rank, args):
    PSNR_list = []
    SSIM_list = []

    # switch to evaluate mode
    model.eval()

    start_t = time.time()
    if local_rank == 0:
        print("local rank {:} begin to validate...".format(local_rank))
    with torch.no_grad():
        for i, (inputSpikes, input_Img, gt_Img, inputIndex) in enumerate(val_loader):
            t1 = time.time()
            inputSpikes = inputSpikes.cuda(local_rank, non_blocking=True)
            input_Img = input_Img.cuda(local_rank, non_blocking=True)
            gt_Img = gt_Img.cuda(local_rank, non_blocking=True)

            ## compute output
            output = model.forward(inputSpikes, input_Img)

            ## measure psnr and ssim
            for res, tar in zip(output, gt_Img):
                psnr = compare_psnr(np.array(tar.to('cpu')), res.to('cpu').detach().numpy())
                ssim = compare_ssim(np.array(tar.permute(1, 2, 0).to('cpu')),
                                    res.permute(1, 2, 0).to('cpu').detach().numpy(), multichannel=True)
                PSNR_list.append(psnr)
                SSIM_list.append(ssim)


            batch_t = time.time() - start_t
            train_t = time.time() - t1
            if local_rank == 0 and i % args.print_freq == 0:
                print('Testing index: {:} Train PSNR: {:.4f}   SSIM: {:.4f}   batch time: {:.2f}   train time: {:.2f}'.format(i, np.mean(PSNR_list), np.mean(SSIM_list), batch_t, train_t))
            start_t = time.time()


        PSNR = np.mean(PSNR_list)
        SSIM = np.mean(SSIM_list)

    return PSNR, SSIM


if __name__ == '__main__':
    main()

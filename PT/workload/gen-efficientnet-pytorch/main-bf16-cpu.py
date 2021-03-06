"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

# from PT.EfficientNet.efficientnet_pytorch.utils import efficientnet_params
import argparse
import os
import random
import shutil
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

params_dict = {
    # Coefficients:   width,depth,res,dropout
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
    'efficientnet_b8': (2.2, 3.6, 672, 0.5),
    'efficientnet_l2': (4.3, 5.3, 800, 0.5),
    'inception_v3': (1, 1, 299, 0.5),
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--ppn', default=1, type=int,
                    help='number of processes on each node of distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex weight cache')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--llga', action='store_true', default=False, help='enable LLGA')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, bfloat16')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('-i', '--iterations', default=200, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("--performance", action='store_true',
                    help="measure performance only, no accuracy.")
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--channels_last', type=int, default=1,
                    help='use channels last format')
parser.add_argument('--config_file', type=str, default="./conf.yaml",
                    help='config file for int8 tune')

args = parser.parse_args()

if args.ipex:
    import intel_pytorch_extension as ipex
    print("Running with IPEX...")
    if args.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)
        print("Running with bfloat16...")

def main():
    args = parser.parse_args()
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.ppn > 1 or args.multiprocessing_distributed

    if args.gpu is not None and args.cuda:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = args.ppn
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        import geffnet
        if args.jit:
            geffnet.config.set_scriptable(True)
        if args.pretrained:
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=True)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=False)
    elif 'mixnet' in args.arch or 'fbnetc_100' in args.arch or 'spnasnet_100' in args.arch:
        import geffnet
        if args.jit:
            geffnet.config.set_scriptable(True)
        if args.pretrained:
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=True)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = geffnet.create_model(args.arch, num_classes=args.num_classes, in_chans=3, pretrained=False)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            if args.arch == "inception_v3":
                model = models.__dict__[args.arch](pretrained=True, aux_logits=False)
            else:
                model = models.__dict__[args.arch](pretrained=True)
        else:
            if args.arch == "inception_v3":
                print("=> creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch](aux_logits=False)
            else:
                print("=> creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch]()
        model.train(True) #kyao
    model.train(True) #kyao
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if args.cuda:
                model.cuda()
                print("create DistributedDataParallel in GPU")
            else:
                print("create DistributedDataParallel in CPU")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        pass
        # # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     if args.cuda:
        #         model.cuda()
        # else:
        #     if not args.jit:
        #         model = torch.nn.DataParallel(model)
        #     if args.cuda:
        #         model.cuda()

    if args.ipex:
        model = model.to(ipex.DEVICE)

    elif args.channels_last:
        model_oob = model
        model_oob = model_oob.to(memory_format=torch.channels_last)
        model = model_oob

    # define loss function (criterion) and optimizer
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train2017')
    valdir = os.path.join(args.data, 'val2017')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if not args.dummy and args.data:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if 'efficientnet' in args.arch:
        image_size = get_image_size(args.arch)
        val_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        args.image_size = image_size
        print('Using image size', image_size)
    elif 'mixnet' in args.arch:
        image_size = 112
        args.image_size = image_size
        print('Using image size', image_size)
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
        print('Using image size', args.image_size)

    val_loader = []
    if not args.dummy and args.data:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, val_transforms),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if args.precision == "int8":
            print("Start quantize model...", end='')
            from lpot.experimental import Quantization, common
            quantizer = Quantization(args.config_file)
            dataset = quantizer.dataset('dummy', (args.batch_size, 3, 224, 224), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = common.Model(model)
            q_model = quantizer()
            model = q_model.model
            print("finished")

        if args.ipex and args.jit:
            try:
                model_ = torch.jit.script(model.eval())
            except:
                try:
                    dummy_inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(ipex.DEVICE)
                    model_ = torch.jit.trace(model.eval(), dummy_inputs)
                except:
                    model_ = model
                    print("Can't convert to jit model...")
        else:
            model_ = model

        # with open('res.txt', 'w') as f:
        #     print(res, file=f)
        #return kyao
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if not args.performance:
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)


def get_image_size(model_name):
    if model_name in params_dict:
        _, _, res, _ = params_dict[model_name]
    else:
        assert False, "Unsupported model:{}".format(model_name)
    return res


def train(train_loader, model, criterion, optimizer, epoch, args):
    print("is training") #kyao
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.iterations > 0 and i >= (args.warmup_iterations + args.iterations):
            break
        # measure data loading time
        if i >= args.warmup_iterations:
            data_time.update(time.time() - end)

        if args.gpu is not None and args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
        if args.cuda:
            target = target.cuda(args.gpu, non_blocking=True)

        elif args.ipex and not args.cuda:
            images = images.to(ipex.DEVICE)

        elif args.channels_last:
            input_oob = images
            input_oob = input_oob.to(memory_format=torch.channels_last)
            images = input_oob

        # compute output
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i >= args.warmup_iterations:
            batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    if args.performance:
        batch_size = train_loader.batch_size
        latency = batch_time.avg / batch_size * 1000
        perf = batch_size/batch_time.avg
        print('training latency: %3.0f ms on %d epoch'%(latency, epoch))
        print('training Throughput: %3.0f fps on %d epoch'%(perf, epoch))





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()
if __name__ == '__main__':
    main()

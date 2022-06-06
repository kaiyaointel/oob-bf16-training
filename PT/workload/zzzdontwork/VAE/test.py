import yaml
import time
import argparse
import os
import torch
from torch import optim
from models import *
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument("--dataset", choices=["celeba"],
                     default="celeba", required=False,
                     help="Kind of dataset",)
parser.add_argument("--model", choices=["VanillaVAE"],
                     default="VanillaVAE", required=False,
                     help="Kind of model",)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=144, type=int,
                    metavar='N', help='batch size (default: 144), this is the total')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)',
                    dest='weight_decay')
parser.add_argument('--sh', '--scheduler-gamma', default=0.95, type=float,
                    metavar='S', help='weight decay (default: 0.95)',
                    dest='scheduler_gamma')
parser.add_argument('--in-channels', default=3, type=int, metavar='N',
                    help='The input chanels')
parser.add_argument('--latent-dim', default=128, type=int, metavar='N',
                    help='The latent-dim')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('-i', '--num-iterations', default=10, type=int, metavar='N',
                    help='number of iterations to run')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--img-size', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex')
parser.add_argument('--jit', action='store_true', default=False,
                    help='use ipex')
parser.add_argument('--precision', default="float32",
                        help='precision, "float32" or "bfloat16"')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--arch', type=str, default=None, help='model name')

args = parser.parse_args()

if args.ipex:
    import intel_pytorch_extension as ipex
    print("Running with IPEX...")
    if args.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print('Running with bfloat16...')

params = {
    'dataset': args.dataset,
    'data_path': args.data,
    'img_size': args.img_size,
    'batch_size': args.batch_size,
    'LR': args.lr,
    'weight_decay': args.weight_decay,
    'scheduler_gamma': args.scheduler_gamma,
}

model_params = {
    'name': args.model,
    'in_channels': args.in_channels,
    'latent_dim': args.latent_dim,
}

config = {
    'model_params': model_params,
    'params': params,
}

model = vae_models[config['model_params']['name']](**config['model_params'])


class TEST_VAE:
    def __init__(self, params, model, args):
        self.args = args
        self.params = params
        self.model = model
        self.test_dataloader()
        self.optimizer = self.configure_optimizers()

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.args.dataset == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def test_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            print("====", self.params['data_path'])
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 self.params['batch_size'],
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            tmp_checkpoint = checkpoint['state_dict']
            for key in list(tmp_checkpoint.keys()):
                if 'model.' in key:
                    tmp_checkpoint[key.replace('model.', '')] = tmp_checkpoint[key]
                    del tmp_checkpoint[key]

            self.model.load_state_dict(tmp_checkpoint)
        else:
            raise ValueError('Checkpoint does not exist')

    def test(self):
        batch_time = AverageMeter('Time', ':6.3f')
        if args.ipex:
            self.device = ipex.DEVICE
        elif self.cuda_enabled:
            torch.cuda.set_device(self.args.gpu)
            self.model.cuda(self.args.gpu)
            self.device = torch.device("cuda:{0}".format(self.args.gpu))
        else:
            self.device = torch.device("cpu")

        if self.args.resume:
            self.load_checkpoint(self.args.resume)
        self.model.eval()
        if self.args.channels_last:
            oob_model = self.model
            oob_model = oob_model.to(memory_format=torch.channels_last)
            self.model = oob_model
            print("---- Use channels last format.")
        else:
            self.model.to(self.device)
        test_loss = 0
        print("===> Running inference:")
        with torch.no_grad():
            for i, (images, target) in enumerate(self.sample_dataloader):
                if self.args.num_iterations != 0 and i >= self.args.num_iterations:
                    break
                if self.args.ipex:
                    images = images.to(ipex.DEVICE)
                    if self.args.jit and i == 0:
                        print("Running jit script model...")
                        self.model = torch.jit.trace(self.model, images)
                elif self.cuda_enabled:
                    images = images.cuda(args.gpu, non_blocking=True)
                elif self.args.channels_last:
                    images_oob = images
                    images_oob = images_oob.to(memory_format=torch.channels_last)
                    images = images_oob

                if i >= self.args.warmup_iterations:
                    end = time.time()

                # compute output
                if self.args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        if self.args.precision == "bfloat16":
                            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                output = self.model(images)
                        else:
                            output = self.model(images)
                else:
                    if self.args.precision == "bfloat16":
                        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            output = self.model(images)
                    else:
                        output = self.model(images)
                # measure elapsed time
                if i >= self.args.warmup_iterations:
                    batch_time.update(time.time() - end)

                if not self.args.jit:
                    test_loss += self.model.loss_function(*output,
                                                M_N = self.params['batch_size'] / self.num_val_imgs,
                                                optimizer_idx=i,
                                                batch_idx = i)['loss'].item()

                if i % self.args.print_freq == 0:
                    print('im_detect: {:d}/{:d}, {:0.3f}({:0.3f}).'.format(i + 1, self.num_val_imgs, batch_time.val, batch_time.avg))

            if not self.args.jit:
                test_loss /= self.num_val_imgs
                print('====> Test AVR loss: {:.4f}'.format(test_loss))

            batch_size = self.sample_dataloader.batch_size
            latency = batch_time.avg / batch_size * 1000
            perf = batch_size / batch_time.avg
            print('inference latency: %0.3f ms' % latency)
            print('inference Throughput: %0.3f fps' % perf)
            #
            if self.args.profile:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    os.makedirs(timeline_dir)
                timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                            "vae" + str(i) + '-' + str(os.getpid()) + '.json'
                print(timeline_file)
                prof.export_chrome_trace(timeline_file)
                # table_res = prof.key_averages().table(sort_by="cpu_time_total")
                # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)

    def save_profile_result(self, filename, table):
        import xlsxwriter
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()
        keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
                "CPU time avg", "Number of Calls"]
        for j in range(len(keys)):
            worksheet.write(0,j,keys[j])

        lines = table.split("\n")
        for i in range(3,len(lines)-4):
            words = lines[i].split(" ")
            j = 0
            for word in words:
                if not word == "":
                    worksheet.write(i-2, j, word)
                    j += 1
        workbook.close()
    @property
    def cuda_enabled(self) -> bool:
         self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
         return self.args.cuda and self.args.gpu is not None

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

test_vae = TEST_VAE(params, model, args)
test_vae.test()

import time
import os
import numpy as np
import simulation
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pytorch_unet

parser = argparse.ArgumentParser(description='PyTorch UNet evaluation')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--n_classes', default=6, type=int,
                    metavar='N', help='the number of class')
parser.add_argument('-i', '--iterations', default=100, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('-w', '--warmup-iterations', default=10, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='use ipex')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--precision', type=str, default="float32",
                    help='precision, float32, int8, bfloat16')
parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')
parser.add_argument('--arch', type=str, help='model name', default="U-Net")
parser.add_argument('--config_file', type=str, default="./conf.yaml", help='config file for int8 tuning')
args = parser.parse_args()

if args.ipex:
    import intel_pytorch_extension as ipex
    print("import IPEX **************")
    if args.precision == "bfloat16":
        # Automatically mix precision
        print("Running with bfloat16...")
        ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16)


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


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)        
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        
        return [image, mask]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def main():
    print(args)

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_set = SimDataset(200, transform = trans)
    dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.ipex:
        device = ipex.DEVICE
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = pytorch_unet.UNet(args.n_classes).eval()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    else:
        model.to(device)
    if args.jit:
        print("Running with jit script model...")
        model = torch.jit.script(model)

    if args.precision == 'int8':
        from lpot.experimental import Quantization, common
        from lpot.utils.pytorch import load
        quantizer = Quantization(args.config_file)
        quantizer.calib_dataloader = dataloader
        quantizer.model = common.Model(model)
        q_model = quantizer()
        model = q_model.model

    batch_time = AverageMeter()
    for i in range(args.iterations):
        inputs, labels = next(iter(dataloader))
        if args.channels_last:
            inputs_oob, labels_oob = inputs, labels
            inputs_oob = inputs_oob.to(memory_format=torch.channels_last)
            labels_oob = labels_oob.to(memory_format=torch.channels_last)
            inputs, labels = inputs_oob, labels_oob
        else:
            inputs = inputs.to(device)
            labels = labels.to(device)
        if i >= args.warmup_iterations:
            start = time.time()
        if args.profile:
            with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
                if args.precision == "bfloat16":
                    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        pred = model(inputs)
                else:
                    pred = model(inputs)
        else:
            if args.precision == "bfloat16":
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    pred = model(inputs)
            else:
                pred = model(inputs)
        
        if i >= args.warmup_iterations:
            batch_time.update(time.time() - start)

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, args.iterations + args.warmup_iterations, batch_time=batch_time))

    latency = batch_time.avg / args.batch_size * 1000
    perf = args.batch_size/batch_time.avg
    print('inference latency: %3.3f ms'%latency)
    print('inference Throughput: %3.3f fps'%perf)

    if args.profile:
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    args.arch + str(i + 1) + '-' + str(os.getpid()) + '.json'

        prof.export_chrome_trace(timeline_file)
        # table_res = prof.key_averages().table(sort_by="cpu_time_total")
        # save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)


if __name__ == '__main__':
    main()

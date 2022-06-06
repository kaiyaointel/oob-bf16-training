from reformer_pytorch import Reformer

import argparse
import time
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset

def add_argument():
    parser=argparse.ArgumentParser(description='enwik8')

    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    parser.add_argument('--ipex', action='store_true', default=False,
                       help='use ipex')
    # parser.add_argument('--jit', action='store_true', default=False,
    #                     help='use ipex')
    parser.add_argument('--precision', default="float32",
                            help='precision, "float32" or "bfloat16"')
    parser.add_argument('--warmup', type=int, default=2,
                        help='number of warmup')
    parser.add_argument('--max_iters', type=int, default=10,
                        help='max number of iterations to run')
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')
    parser.add_argument('--arch', type=str, help='model name.')

    args=parser.parse_args()
    return args


# instantiate model

# model = ReformerLM(
#     dim = 512,
#     depth = 6,
#     max_seq_len = SEQ_LEN,
#     num_tokens = 256,
#     heads = 8,
#     bucket_size = 64,
#     n_hashes = 4,
#     ff_chunks = 10,
#     lsh_dropout = 0.1,
#     weight_tie = True,
#     causal = True,
#     n_local_attn_heads = 4,
#     use_full_attn = False # set this to true for comparison with full attention
# )

# model = TrainingWrapper(model)
print("create model...")
model = Reformer(
    dim = 512,
    depth = 12,
    max_seq_len = 4096,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True
)
# model.cuda()

# prepare enwik8 data


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


def create_dataset():
    with gzip.open('./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg
# setup deepspeed

cmd_args = add_argument()
input = torch.randn(cmd_args.batch_size, 4096, 512)
if cmd_args.ipex:
    import intel_pytorch_extension as ipex
    print("Running with IPEX...")
    if cmd_args.precision == "bfloat16":
        # Automatically mix precision
        ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        print('Running with bfloat16...')
    input = input.to(ipex.DEVICE)
elif cmd_args.with_cuda:
    input = input.cuda()
elif cmd_args.channels_last:
    try:
        input_oob = input
        input_oob = input_oob.to(memory_format=torch.channels_last)
        input = input_oob
    except:
        print("Input NHWC failed! Use normal input.")

model.eval()
if cmd_args.ipex:
    model.to(ipex.DEVICE)
    # if cmd_args.jit:
    #     model = torch.jit.trace(model, input)
elif cmd_args.with_cuda:
    model.cuda()
elif cmd_args.channels_last:
    try:
        model_oob = model
        model_oob = model_oob.to(memory_format=torch.channels_last)
        model = model_oob
    except:
        print("Model NHWC failed! Use normal input.")

# training

batch_time = AverageMeter()
with torch.no_grad():
    for i in range(cmd_args.max_iters + cmd_args.warmup):
        if i >= cmd_args.warmup:
            start = time.time()
            if cmd_args.profile:
                with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
                    if cmd_args.precision == "bfloat16":
                        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            output = model(input)
                    else:
                        output = model(input)
            else:
                if cmd_args.precision == "bfloat16":
                    with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        output = model(input)
                else:
                    output = model(input)
            batch_time.update(time.time() - start)
        else:
            output = model(input)
        # if i % 10 == 0:
        print("iterations:{}/({})".format(i+1, cmd_args.max_iters + cmd_args.warmup))

    latency = batch_time.avg / cmd_args.batch_size * 1000
    perf = cmd_args.batch_size / batch_time.avg
    print('inference latency: %0.3f ms' % latency)
    print('inference Throughput: %0.3f samples/s' % perf)

    if cmd_args.profile:
        import pathlib
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    cmd_args.arch + str(i + 1) + '-' + str(os.getpid()) + '.json'
        prof.export_chrome_trace(timeline_file)


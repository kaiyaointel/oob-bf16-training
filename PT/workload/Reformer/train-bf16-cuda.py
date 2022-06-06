from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import time #kyao
import pathlib
import os

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 4096

class AverageMeter(object):  #kyao (whole class)
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

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate model

model = ReformerLM(
    dim = 512,
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True,
    n_local_attn_heads = 4,
    use_full_attn = False # set this to true for comparison with full attention
)

model = TrainingWrapper(model)
model.cuda()

# prepare enwik8 data

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
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

batch_time = AverageMeter('Time', ':6.3f') #kyao
end = time.time() #kyao

count = 0
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        count = count + 1
        print("count = ", count)
        if count == 5:
            exit()
        with torch.profiler.profile(
            activities = [torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            with torch.cuda.amp.autocast(enabled=True):
                loss = model(next(train_loader), return_loss = True)
                loss.backward()
                
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    str(i + 1) + '-' + str(os.getpid()) + '.json'

        prof.export_chrome_trace(timeline_file)

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    
    batch_time.update(time.time() - end) #kyao
    end = time.time() #kyao
    
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    #if i % GENERATE_EVERY == 0:
    #    model.eval()
    #    inp = random.choice(val_dataset)[:-1]
    #    prime = decode_tokens(inp)
    #    print(f'%s \n\n %s', (prime, '*' * 100))
    #
    #    sample = model.generate(inp, GENERATE_LENGTH)
    #    output_str = decode_tokens(sample)
    #    print(output_str)
        
    ### performance computation #kyao
    latency = batch_time.avg / BATCH_SIZE * 1000 #kyao
    throughput = BATCH_SIZE / batch_time.avg #kyao
    print('training latency: %.3f ms on %d epoch'%(latency, i)) #kyao
    print('training throughput: %.3f fps on %d epoch'%(throughput, i)) #kyao

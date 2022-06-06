import pandas as pd
import torch
import os
import time
import argparse
from tqdm import trange
from pytorch_widedeep.preprocessing import WidePreprocessor, DeepPreprocessor
from pytorch_widedeep.models import Wide, DeepDense, WideDeep
from pytorch_widedeep.metrics import BinaryAccuracy
from pytorch_widedeep.models._wd_dataset import WideDeepDataset

from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--num_warmup', type=int, default=5, help='input batch size')
parser.add_argument('--num_workers', type=int, default=1, help='input batch size')
parser.add_argument('--num_iter', type=int, default=0, help='input batch size')
parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--pretrained', default='./model/wide_deep.pth', help="path to pretrained model")
parser.add_argument('--save_dir', default='./model/', help='Where to store models')
parser.add_argument('--inf', action='store_true', help='inference only')
parser.add_argument('--ipex', action='store_true', default=False, help='Use ipex to get boost.')
# parser.add_argument('--jit', action='store_true', default=False,
#                      help='enable Intel_PyTorch_Extension JIT path')
parser.add_argument('--precision', default='float32', help='precision, "float32" or "bfloat16"')
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--channels_last', type=int, default=1,
                    help='use channels last format')
parser.add_argument('--config_file', type=str, default="./conf.yaml",
                    help='config file for int8 tuning')
args = parser.parse_args()
print(args)


class _DataLoader(object):
    def __init__(self, dataset, batch_size=1):
        self.data = dataset
        self.batch_size = batch_size
    def __iter__(self):
        for data in self.data:
            yield (data,), "l"

def dataProcess():
    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income'
    ]
    # these next 3 lines are not directly related to pytorch-widedeep. I assume
    # you have downloaded the dataset and place it in a dir called data/adult/
    df = pd.read_csv('data/adult.data', names=_CSV_COLUMNS)

    df['income_label'] = (df["income"].apply(lambda x: ">50K" in x)).astype(int)
    df.drop('income', axis=1, inplace=True)
    return df

def _get_wide_deep():
    df = dataProcess()
    # prepare wide, crossed, embedding and continuous columns
    wide_cols  = ['education', 'relationship', 'workclass', 'occupation','native-country', 'gender']
    cross_cols = [('education', 'occupation'), ('native-country', 'occupation')]
    embed_cols = [('education',16), ('workclass',16), ('occupation',16),('native-country',32)]
    cont_cols  = ["age", "hours-per-week"]
    target_col = 'income_label'

    # target
    target = df[target_col].values
    # wide
    preprocess_wide = WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
    X_wide = preprocess_wide.fit_transform(df)
    wide = Wide(wide_dim=X_wide.shape[1], output_dim=1)

    # deepdense
    preprocess_deep = DeepPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
    X_deep = preprocess_deep.fit_transform(df)
    deepdense = DeepDense(hidden_layers=[64,32],
                        deep_column_idx=preprocess_deep.deep_column_idx,
                        embed_input=preprocess_deep.embeddings_input,
                        continuous_cols=cont_cols)


    return X_wide, wide, X_deep, deepdense, target

def train():

    X_wide, wide, X_deep, deepdense, target = _get_wide_deep()

    # build, compile, fit and predict
    model = WideDeep(wide=wide, deepdense=deepdense)
    model.compile(method='binary', metrics=[BinaryAccuracy])
    model.fit(X_wide=X_wide, X_deep=X_deep, target=target, n_epochs=args.nepoch, batch_size=args.batch_size, val_split=0.2)
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_model = save_path + "wide_deep.pth"
    torch.save(model, save_model)

def inference():
    X_wide, wide, X_deep, deepdense, _ = _get_wide_deep()
    load_dict = {"X_wide": X_wide, "X_deep": X_deep}
    test_set = WideDeepDataset(**load_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_steps = (len(test_loader.dataset) // test_loader.batch_size) + 1
    model = torch.load(args.pretrained)

    if args.ipex:
        import intel_pytorch_extension as ipex
        print("Running with IPEX...")
        if args.precision == "bfloat16":
            # Automatically mix precision
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
            print("Running with bfloat16...")
        model = model.to(ipex.DEVICE)

    if args.channels_last:
        model_oob = model
        model = model.to(memory_format=torch.channels_last)
        model = model_oob
        print("---- Use channels last format.")

    if args.precision == 'int8':
        from lpot.experimental import Quantization, common
        quantizer = Quantization(args.config_file)
        quantizer.calib_dataloader = _DataLoader(test_loader)
        quantizer.model = common.Model(model)
        q_model = quantizer()
        model = q_model.model

    # if args.jit:
    #     # model = torch.jit.trace(model.eval(), X_wide, X_deep)
    #     model = torch.jit.script(model.eval())

    totle_time = 0
    iter_i = 0
    with torch.no_grad():
        with trange(test_steps, disable=False) as t:
            for i, data in zip(t, test_loader):
                if args.num_iter > 0 and i >= args.num_iter:
                    break
                if i >= args.num_warmup:
                    tic = time.time()

                t.set_description("predict")
                if args.ipex:
                    X = {k: v.to(ipex.DEVICE).float() for k, v in data.items()}
                else:
                    X = {k: v.float() for k, v in data.items()}
                if args.channels_last:
                    oob_X = X
                    oob_X = {k: v.to(memory_format=torch.channels_last) for k, v in data.items()}
                    X = oob_X

                if args.profile:
                    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
                        if args.precision == "bfloat16":
                            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                preds = torch.sigmoid(model.forward(X))
                        else:
                            preds = torch.sigmoid(model.forward(X))
                else:
                    if args.precision == "bfloat16":
                        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                            preds = torch.sigmoid(model.forward(X))
                    else:
                        preds = torch.sigmoid(model.forward(X))
                        # preds = model._activation_fn(model.forward(X))
                # if model.method == "multiclass":
                #     preds = F.softmax(preds, dim=1)
                preds = preds.cpu().data.numpy()
                if i >= args.num_warmup:
                    iter_i += 1
                    totle_time += time.time() - tic
            
            if args.profile:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    os.makedirs(timeline_dir)
                timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                            "wide_deep" + str(i + 1) + '-' + str(os.getpid()) + '.json'

                prof.export_chrome_trace(timeline_file)
                # table_res = prof.key_averages().table(sort_by="cpu_time_total")
                # save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)

            throughput = iter_i * test_loader.batch_size / totle_time
            print("Latency: %s ms" % str(1000 / throughput))
            print("Throughput: %s samples/sec" % str(throughput))

    return

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
                time.sleep(0.1)
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()

def main():
    if args.eval:
        inference()
    else:
        train()


if __name__ == '__main__':
    main()


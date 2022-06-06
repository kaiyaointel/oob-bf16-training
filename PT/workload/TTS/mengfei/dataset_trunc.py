import argparse
import csv
import sys
import statistics
import random

def main():
    args = parse_args()
    truncated_num = args.n
    original_list = load_test_sentences(args.dataset_name, args.metadata_path)
    retries = args.retries
    output_path = args.output
    if len(original_list) < truncated_num:
        print(f"[INFO] Original list already shorter than {truncated_num} sentences. Exiting.")
        return
    truncated_list = generate_truncated_list(retries, original_list, truncated_num,
                                             args.dataset_name, args.tolerate,
                                             args.mean_limit, args.median_limit, args.stdev_limit)
    if truncated_list:
        save_test_sentences(output_path, truncated_list, args.dataset_name)

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", help="Number of elements in truncated data", action="store", type=int, required=True)
    parser.add_argument("--dataset-name", help="Dataset name", action="store", default="dummy", required=False)
    parser.add_argument("--metadata-path", help="Dataset metadata filename (file containing sentences)",
                        default="./dummy_data.csv", action="store", required=False)
    parser.add_argument("--mean-limit", help="Acceptable mean difference percentage", action="store", default=1, type=int, required=False)
    parser.add_argument("--median-limit", help="Acceptable median difference percentage", action="store", default=1, type=int, required=False)
    parser.add_argument("--stdev-limit", help="Acceptable stdev difference percentage", action="store", default=1, type=int, required=False)
    parser.add_argument("--tolerate", help="Saves truncated dataset regardles of differences", action="store_true", required=False)
    parser.add_argument("--retries", help="Number of max retries", action="store", default=10, type=int, required=False)
    parser.add_argument("--output", "-o", help="Output path", action="store", default="./truncated_metadata.csv", required=False)
    args = parser.parse_args()
    return args

def load_test_sentences(dataset_name, metadata_path):
    dataset_load_function = {
        "dummy": get_dummy_data,
        "ljspeech": get_ljspeech_data
    }
    with open(metadata_path) as filestream:
        raw_data = filestream.readlines()
    return dataset_load_function.get(dataset_name, lambda x: [])(raw_data)

def get_dummy_data(raw_data):
    reader = csv.reader(raw_data)
    data = [row for row in reader if row]
    return data

def get_ljspeech_data(raw_data):
    reader = csv.reader(raw_data, delimiter="|", quoting=csv.QUOTE_NONE)
    data = [row for row in reader]
    return data

def generate_truncated_list(retries, original_list, truncated_num, dataset_name,
                    tolerate, mean_limit, median_limit, stdev_limit):
    for i in range(retries):
        truncated_list = random.sample(original_list, truncated_num)
        original_sentences_lengths = get_sentences_lengths(dataset_name, original_list)
        truncated_sentences_lengths = get_sentences_lengths(dataset_name, truncated_list)
        diff_mean, diff_median, diff_stdev = sentence_length_statistics(original_sentences_lengths,
                                                                        truncated_sentences_lengths)
        if tolerate or check_difference(diff_mean, mean_limit,
                                             diff_median, median_limit,
                                             diff_stdev, stdev_limit):
            print(f"[INFO] Acceptable subset generated.")
            return truncated_list
    print(f"[INFO] Unable to generate acceptable subset after {retries} retries.")
    return None

def sentence_length_statistics(original_sentences_lengths, truncated_sentences_lengths):
    orig_mean = statistics.mean(original_sentences_lengths)
    orig_median = statistics.median(original_sentences_lengths)
    orig_stdev = statistics.stdev(original_sentences_lengths)
    trunc_mean = statistics.mean(truncated_sentences_lengths)
    trunc_median = statistics.median(truncated_sentences_lengths)
    trunc_stdev = statistics.stdev(truncated_sentences_lengths)
    diff_mean = abs(100 - trunc_mean/orig_mean*100)
    diff_median = abs(100 - trunc_median/orig_median*100)
    diff_stdev = abs(100 - trunc_stdev/orig_stdev*100)
    print("[INFO] Original vs truncated dataset comparison:")
    print(f"[INFO] Original mean: {orig_mean} | Truncated mean: {trunc_mean} | Difference: {diff_mean:.2f}%")
    print(f"[INFO] Original median: {orig_median} | Truncated median: {trunc_median} | Difference: {diff_median:.2f}%")
    print(f"[INFO] Original stdev: {orig_stdev} | Truncated stdev: {trunc_stdev} | Difference: {diff_stdev:.2f}%")
    return diff_mean, diff_median, diff_stdev

def get_sentences_lengths(dataset_name, data):
    counter_lambdas = {
        "dummy": lambda x: len(x),
        "ljspeech": lambda x: len(x[2])
    }
    lamb = counter_lambdas.get(dataset_name, lambda x: None)
    original_sentences_lengths = list(map(lamb, data))
    return original_sentences_lengths

def check_difference(diff_mean, mean_limit, diff_median, median_limit, diff_stdev, stdev_limit):
    if diff_mean > mean_limit:
        print(f"[INFO] Mean difference {diff_mean}% not within {mean_limit}% limit.")
        return False
    if diff_median > median_limit:
        print(f"[INFO] Median difference {diff_median}% not within {median_limit}% limit.")
        return False
    if diff_stdev > stdev_limit:
        print(f"[INFO] Stdev difference {diff_stdev}% not within {stdev_limit}% limit.")
        return False
    print(f"[INFO] All statistics within limits.")
    return True

def save_test_sentences(metadata_path, data, dataset_name):
    dataset_dialects = {
        "dummy": {
            "delimiter": "\n"
        },
        "ljspeech": {
            "delimiter": "|",
            "quoting": csv.QUOTE_NONE,
            "quotechar": ""
        }
    }
    dialect = dataset_dialects.get(dataset_name, {})
    with open(metadata_path, "w") as filestream:
        writer = csv.writer(filestream, **dialect)
        writer.writerows(data)

def sniff_dialect(metadata_path):
    filestream = open(metadata_path)
    dialect = csv.Sniffer().sniff(filestream.read())
    filestream.close()
    return dialect

if __name__ == "__main__":
    sys.exit(main())

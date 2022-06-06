import os
import json
import argparse

def parse_file(file_path, display_op):
    data = ''
    try:
        # Reading data back
        with open(file_path, 'r') as f:
            data = json.load(f)['traceEvents']

    except Exception as e:
        print("Profiling json file parse ERROr! ")

    if data is not None:
        Op_profile_detail = {}
        for detail_dict in data:
            if 'cat' in detail_dict.keys() and detail_dict['cat'] == 'Op':
                op_name = detail_dict['name']
                op_time = detail_dict['dur']
                if op_name in Op_profile_detail.keys():
                    Op_profile_detail[op_name] += op_time
                else:
                    Op_profile_detail[op_name] = op_time

    Op_profile_detail = sorted(Op_profile_detail.items(), key=lambda d: d[1], reverse=True)
    if display_op is not None:
        for item in Op_profile_detail:
            if display_op == item[0]:
                print("{}: {} ms".format(item[0], item[1]/1000))
            return item

    else:
        for item in Op_profile_detail:
            print("{}: {} ms".format(item[0], item[1]/1000))
        return Op_profile_detail

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="path of profiling json file")
    parser.add_argument("--display_op", help="op name to display")

    args = parser.parse_args()
    parse_file(args.file_path, args.display_op)
    
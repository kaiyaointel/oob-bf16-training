import os
import json
import argparse


class Profile:
    def __init__(self, name, time, calls):
        self.name = name
        self.time = time
        self.calls = calls
    def __repr__(self):
        return repr((self.name, self.time, self.age))


def Parse_File(file_path, display_op):
    # load json file
    data = ''
    try:
        # Reading data back
        with open(file_path, 'r') as f:
            data = json.load(f)['traceEvents']
    except Exception as e:
        print("Profiling json file parse ERROr! ")

    # generate op detials
    op_list = []
    max_name_len = 0
    total_time = 0
    if data is not None:
        for detail_dict in data:
            if 'dur' in detail_dict.keys():
                op_name = str(detail_dict['name'])
                op_time = int(detail_dict['dur']) / 1000
                op_calls = 1
                total_time += op_time

                if display_op is not None:
                    if op_name != display_op:
                        continue

                if len(op_name) > max_name_len:
                    max_name_len = len(op_name)
                
                if op_name not in [op.name for op in op_list]:
                    op_list.append(Profile(op_name, op_time, op_calls))
                else:
                    for index, op in enumerate(op_list):
                        if op_name == op.name:
                            op_list[index].time += op_time
                            op_list[index].calls += op_calls
    # sort
    op_list = sorted(op_list, key=lambda op: op.time, reverse=True)

    # print op
    print("Operator" + " "*(max_name_len - 8) + "\tDuration(ms)\tOcurrences\tPercentage(%)")
    for op in op_list:
        col1 = op.name + " "*(max_name_len - len(op.name))
        col2 = op.time
        col3 = op.calls
        col4 = op.time / total_time * 100
        print("{}\t{:.3f}\t\t{}\t\t{:.0f}".format(col1, col2, col3, col4))
    print("Total Time: %s ms for %d items.\n\n" % (total_time, len(op_list)))

    return op_list


if __name__ == "__main__":
    # parar args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, default="", help="path of profiling json file")
    parser.add_argument("-p", "--display_op", type=str, default=None, help="op name to display")
    args = parser.parse_args()

    Parse_File(args.file_path, args.display_op)

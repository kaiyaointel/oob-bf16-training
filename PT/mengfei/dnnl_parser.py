
import os
import argparse


class Verbose:
    def __init__(self, name, time, calls):
        self.name = name
        self.time = time
        self.calls = calls
    def __repr__(self):
        return repr((self.name, self.time, self.age))

def Parse_File(file_path='verbose.log', config=False, primitive_display=None):
    primitive_list = []
    max_name_len = 0
    total_time = 0

    with open(file_path, 'r') as f:
        for line in f:
            if len(line) > 1 and (line.startswith('mkldnn_verbose') or line.startswith('dnnl_verbose')):
                # skip the version line sicne 0.18
                if 'info' in line or 'create' in line:
                    continue

                primitive, shape, time = Parse_Line(line)
                if 'backward_data' in line:
                    primitive = primitive + '_backward_data'
                elif 'backward_weights' in line:
                    primitive = primitive + '_backward_weights'
                elif 'backward' in line:
                    primitive = primitive + '_backward'
                if config:
                    primitive = shape
                total_time += time

                if primitive_display is not None:
                    if primitive != primitive_display:
                        continue

                if len(primitive) > max_name_len:
                    max_name_len = len(primitive)

                if primitive in [prim.name for prim in primitive_list]:
                    for index, prim in enumerate(primitive_list):
                        if primitive == prim.name:
                            primitive_list[index].time += time
                            primitive_list[index].calls += 1
                else:
                    primitive_list.append(Verbose(primitive, time, 1))

    return primitive_list, max_name_len, total_time

def Parse_Line(line):
    assert len(line) > 1
    if line.startswith('mkldnn_verbose'):
        p_pos = 2
    elif line.startswith('dnnl_verbose'):
        p_pos = 3
    else:
        assert False, "verbose line should start with mkldnn_verbose or dnnl_verbose"

    p = line[:-1].split(',')
    primitive = p[p_pos]
    shape = p[-2]
    time = float(p[-1])

    return primitive, shape, time

def Print_Title(max_name_len, config=False):
    col1 = 'Shape' if config else 'Primitive'
    col1 = col1 + " "*(max_name_len - len(col1))
    col2 = 'Duration (ms)'
    col3 = 'Occurrences'
    col4 = 'Percentage (%)'
    print("{}\t{}\t{}\t{}".format(col1, col2, col3, col4))

def Print_All(primitive_list, max_name_len, total_time, config=False):
    # sort
    primitive_list = sorted(primitive_list, key=lambda prim: prim.time, reverse=True)

    Print_Title(max_name_len, config)
    for prim in primitive_list:
        col1 = prim.name + " "*(max_name_len - len(prim.name))
        col2 = prim.time
        col3 = prim.calls
        col4 = prim.time / total_time * 100
        print("{}\t{:.3f}\t\t{}\t\t{:.0f}".format(col1, col2, col3, col4))

    print("Total Time: %s ms for %d items.\n\n" % (total_time, len(primitive_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mkl-dnn verbose log analysis",             # noqa
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # noqa
    parser.add_argument("-f", "--file_path",
                        help="mkl-dnn verbose log file", default=None)
    parser.add_argument("-p", "--primitive_display",
                        help="primitive need to analyze", default=None)
    parser.add_argument("-c", "--config",
                        action='store_true',
                        help="statistics of each configuration for a specific primitive")
    args = parser.parse_args()

    # parser file
    primitive_list, max_name_len, total_time = Parse_File(args.file_path, config=args.config, primitive_display=args.primitive_display)

    # print
    Print_All(primitive_list, max_name_len, total_time, config=args.config)


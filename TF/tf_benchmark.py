import os
import sys
import time
import logging
import argparse

from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.protobuf import rewriter_config_pb2

from find_outputs import get_input_output
from utils import *

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOB-Benchmark")


if "PRINT_LATENCY" in os.environ and int(os.environ["PRINT_LATENCY"]) == 1:
    PRINT_LATENCY = True
else:
    PRINT_LATENCY = False

if "RUN_PROFILING" in os.environ and int(os.environ["RUN_PROFILING"]) == 1:
    RUN_PROFILING = True
else:
    RUN_PROFILING = False

if "SAVE_RUNTIME_GRAPH" in os.environ and int(os.environ["SAVE_RUNTIME_GRAPH"]) == 1:
    SAVE_RUNTIME_GRAPH = True
else:
    SAVE_RUNTIME_GRAPH = False

if "CHECK_ACCURACY" in os.environ and os.environ["CHECK_ACCURACY"] == "1":
    CHECK_ACCURACY = True
else:
    CHECK_ACCURACY = False

if "BS" in os.environ:
    BATCH_SIZE = int(os.environ["BS"])
else:
    BATCH_SIZE = 1


if "OMP_NUM_THREADS" in os.environ:
    NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
else:
    NUM_THREADS = 28

def metrics_generator(array, tolerance):
    max_diff = np.max(array)
    mean_diff = np.mean(array)
    median_diff = np.median(array)
    success_rate = np.sum(array < tolerance) / array.size
    return max_diff, mean_diff, median_diff, success_rate


def initialize_graph(model_details, disable_optimize_for_inference):
    graph = tf_v1.Graph()
    with graph.as_default():
        input_variables = {
            in_name + ":0": tf_v1.Variable(val)
            for in_name, val in model_details['input'].items()}

        od_graph_def = tf_v1.GraphDef()
        with tf_v1.gfile.GFile(os.path.join(os.getcwd(), model_details['model_dir']), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            od_graph_def = delete_assign(od_graph_def)
        
        # optimize for inference
        if not disable_optimize_for_inference:
            # optimize graph for inference
            input_list = [in_name for in_name,
                            val in model_details['input'].items()]
            output_list = [
                out_name for out_name in model_details['output']]
            input_data_type = [tf_v1.convert_to_tensor(item).dtype.as_datatype_enum for item in model_details['input'].values()]

            od_graph_def = optimize_for_inference_lib.optimize_for_inference(
                od_graph_def,  # inputGraph,
                input_list,  # an array of the input nodes
                output_list,  # an array of output nodes
                input_data_type)

        tf_v1.import_graph_def(od_graph_def, name='g',
                            input_map=input_variables)

    return graph


def create_tf_config(args):
    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = NUM_THREADS
    config.inter_op_parallelism_threads = 1
    if args.precision == 'bfloat16':
        config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
    return config


def run_benchmark(model_details, max_reps, num_warmup, args):
    tf_config = create_tf_config(args)
    graph = initialize_graph(model_details, args.disable_optimize)
    run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
    run_metadata = tf_v1.RunMetadata()
   
    if SAVE_RUNTIME_GRAPH:
        # write the real benchmark graph to local
        model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
        out_graph_file = os.path.join(model_dir, 'runtime_graph.pb')
        write_graph(graph.as_graph_def(), out_graph_file)
        print("********** save runtime graph at {}".format(out_graph_file))
 
    with tf_v1.Session(config=tf_config, graph=graph) as sess:
        output_dict = {out_name: graph.get_tensor_by_name("g/" + out_name + ":0")
                       for out_name in model_details['output']}

        sess.run(tf_v1.global_variables_initializer())

        total_time = 0.0
        reps_done = 0
        for rep in range(max_reps):
            if rep < num_warmup:
                _ = sess.run(output_dict)
                continue
            start = time.time()
            if RUN_PROFILING:
                _ = sess.run(output_dict, options=run_options,
                             run_metadata=run_metadata)
            else:
                _ = sess.run(output_dict)

            if RUN_PROFILING:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                model_dir = os.path.dirname(os.path.abspath(model_detail['model_dir']))
                if not os.path.exists(model_dir + '/timeline'):
                    os.makedirs(model_dir + '/timeline')
                with open(model_dir + '/timeline/' + 'timeline_' + str(rep + 1) + '.json', 'w') as trace_file:
                    trace_file.write(
                        trace.generate_chrome_trace_format(show_memory=False))

            end = time.time()
            delta = end - start
            total_time += delta
            reps_done += 1

        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * BATCH_SIZE
        if PRINT_LATENCY:
            print("Latency: {:.0f} ms".format(latency))
        else:
            print("Throughput: {:.2f} fps".format(throughput))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--model_path", help="path of model")
    parser.add_argument("--model_name", help="name of model")
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_true',
                        help="use this to disable optimize_for_inference")
    parser.add_argument("--is_meta", action='store_true',
                        help="input a meta file")
    parser.add_argument("--precision", type=str, default='float32',
                        help="float32 or int8 or bfloat16")

    args = parser.parse_args()

    num_iter = args.num_iter
    num_warmup = args.num_warmup
    # model_path = args.model_path
    is_meta = args.is_meta

    # benchmark PB model directly
    if args.model_path and not args.model_name:
        # generate model detail
        model_dir = args.model_path
        model_detail = {}
        model_input_output = get_input_output(model_dir, is_meta)
        # ckpt/meta model will save freezed pb in the same dir
        model_dir = model_dir if not is_meta else args.model_path[:-5] + "_freeze.pb"
        output = model_input_output['outputs']
        input_dic = {}
        for _input in model_input_output['inputs']:
            # deal with bool dtype input
            if model_input_output['inputs'][_input]['type'] == 'bool':
                input_dic[_input] = model_input_output['inputs'][_input]['value']
                logger.info("Find benchmark input name: {}, dtype: {}".format(_input, model_input_output['inputs'][_input]['type']))
            elif _input == 'dropout_keep_prob':
                input_dic[_input] = np.array([0.5,], dtype='float32')
            else:
                dtype = model_input_output['inputs'][_input]['type']
                dshape = model_input_output['inputs'][_input]['shape']
                dummy_input = generate_data(dshape, dtype, BATCH_SIZE)
                input_dic[_input] = dummy_input
                logger.info("Find benchmark input name: {}, dtype: {}, shape: {}"
                            .format(_input, dtype, dummy_input.shape))
        model_detail['model_dir'] = model_dir
        model_detail['input'] = input_dic
        model_detail['output'] = output
        model_detail['ckpt'] = args.is_meta


    # benchmark with input/output
    elif args.model_name:
        assert args.model_path is not None, "Model path is undefined."
        from model_detail import models
        model_detail = None
        for model in models:
            if model['model_name'] == args.model_name:
                model_detail = model
                model_detail['model_dir'] = args.model_path
                model_detail['ckpt'] = args.is_meta
                break

        if not model_detail:
            logger.error("Model undefined.")
            sys.exit(1)
    
    input_shapes = []
    for input_tensor in model_detail['input'].values():
        if not isinstance(input_tensor, bool):
            input_shapes.append(list(input_tensor.shape))
        else:
            input_shapes.append(['bool'])

    logger.info("***** Final benchmark input name: {}, shape: {}".format( \
                model_detail['input'].keys(), input_shapes))
    logger.info("***** Final benchmark output name: {}".format(model_detail['output']))

    if RUN_PROFILING:
        logger.info("Do profiling, set num_iter to 20, and 10/20 for warmup.")
        num_warmup = 10
        num_iter = 20

    run_benchmark(model_detail, num_iter, num_warmup, args)

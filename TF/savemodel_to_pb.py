import os
import argparse
import tensorflow as tf

def savemodel_valid(meta_graph):
    valid_op=["Conv2D","DepthwiseConv2dNative","MaxPool","AvgPool","FusedBatchNorm","FusedBatchNormV3","BatchNormWithGlobalNormalization",
                 "Relu","Relu6","Softmax","BiasAdd","Add","AddV2"]
    all_op_types = []
    for i in meta_graph.graph_def.node:
        all_op_types.append(i.op)
    print (set(all_op_types))
    flag=False
    for op in set(all_op_types):
        if op in valid_op:
           flag=True
    return flag
def savemodel2pb(model_dir,save_path):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        meta_graph=tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],model_dir)
        assert savemodel_valid(meta_graph),"savemodel is not valid"
        model_graph_signature = list(meta_graph.signature_def.items())[0][1]
        output_tensor_names = []
        for output_item in model_graph_signature.outputs.items():
            output_tensor_name = output_item[1].name
            output_tensor_names.append(output_tensor_name)
        output_node_lists=[output_name.split(":")[0] for output_name in output_tensor_names]
        output_node_names=""
        for out_name in output_node_lists:
            if output_node_names=="":
               output_node_names+=out_name
            else:
               output_node_names=output_node_names+','+out_name
        print("output node names are:%s"%output_node_names)
        os.system("python3 -m tensorflow.python.tools.freeze_graph --input_saved_model_dir=%s --output_node_names=%s  --output_graph=%s"%(model_dir,output_node_names,save_path))
        print("savemodel to pb done")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",help="path of savemodel", required=True)
    parser.add_argument("-s", "--save_path",help="save path of pb", required=True)
    args = parser.parse_args()

    savemodel2pb(args.model_path,args.save_path)

# from keras.models import load_model
from keras.models import model_from_json
import json
import tensorflow as tf
import os 
import os.path as osp
# from keras import backend as K
from tensorflow.compat.v1.keras import backend as K
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
# from tensorflow.keras import activations, initializers, constraints, regularizers
# from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape
from keras.utils import CustomObjectScope
from misc import SqueezedSparseConversion,GraphConvolution,GatherIndices

input_path = os.path.abspath('models/')
weight_file = 'gcn.h5'
weight_file_path = os.path.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'


# def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    #out_nodes.append(h5_model.output[1])
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()  
    from tensorflow.python.framework import graph_util,graph_io
    print("output nodes is : ",out_nodes)
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

output_dir = input_path #osp.join(os.getcwd(),"trans_model")
custom_ob = {'SqueezedSparseConversion': SqueezedSparseConversion, 'GraphConvolution': GraphConvolution, 'GatherIndices': GatherIndices}
h5_model = load_model(weight_file_path, custom_objects=custom_ob)
h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
print('model saved')

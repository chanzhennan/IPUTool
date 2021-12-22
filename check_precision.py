#!/usr/bin/python
#****************************************************************#
# ScriptName: load_uc_model.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-12-16 14:24
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-12-20 10:38
# Function:
#***************************************************************#
import os
from statistics import mean
import time
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
from collections import defaultdict
from tensorflow.python import ipu

# Builds ipu_options
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.configure_ipu_system()
np.random.seed(1991)
ops.disable_eager_execution()
tf.disable_v2_behavior()


global tag 
tag = None

def gen_data(model_path, bs = 1):
    with open(f"{model_path}/raw_data.dat") as dat_file:
        dat_content = dat_file.read().strip().split('[dat]')

    input_str_list = []
    for s in dat_content:
      if s:
        s = '[dat]' + s
        input_str_list.append(s)

    input_strs = [ "".join(input_str_list[i: i + bs]) for i in range(0, len(dat_content), bs) if "".join(input_str_list[i: i + bs])]

    return input_strs

def get_output_tensor_name(path):
  with tf.Session(graph=tf.Graph()) as sess:
    meta = tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, path)
    return [ i.name for i in meta.signature_def["serving_default"].outputs.values()]

def pb2tensor(path, data, output_list, _feed_dict=None):
    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)
    os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --show_progress_bar=true'

    with tf.Session(graph=tf.Graph()) as sess:
        meta = tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, path)
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_list]
        if _feed_dict:
          o = sess.run(out_names_pl, feed_dict=_feed_dict)
        else:
          o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (data,)})
    return o

def compare_model(path1, path2, data):
    sess_cfg = tf.ConfigProto()
    #sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (rewriter_config_pb2.RewriterConfig.OFF)

   
    output_list = get_output_tensor_name(path1, tag)
    output_list2 = ['concat:0',
    'query_attenqt_cls/dense/Relu/OutputAddition0:0',
    'query_attenqc_cls/dense/Relu/OutputAddition0:0',]

    output_list.sort()
    output_list2.sort()
    print(output_list2)
    print(output_list2)

    data1 = pb2tensor(path1, data, output_list, tag)
    data2 = pb2tensor(path2, data, output_list2, tag)
    return data1, data2

def get_tensor(path, data):
    name_list = ['TensorDict/StandardKvParser:0', 'TensorDict/StandardKvParser:10', 'TensorDict/StandardKvParse:16', 'TensorDict/StandardKvParser:4', 'TensorDict/StandardKvParser:9']
    d = run_model(path, data, name_list, tag)
    return

def run_embedruntime(path1, path2, data):
    list1 = ['concat:0', 'query_attenqt_cls/dense/Relu:0', 'query_attenqc_cls/dense/Relu:0',]
    list2 = ['TensorDict/StandardKvParser:0', 'TensorDict/StandardKvParser:10', 'TensorDict/StandardKvParser:16', 'TensorDict/StandardKvParser:4', 'TensorDict/StandardKvParser:9']
    list3 = ['TensorDict/StandardKvParser:0', 'TensorDict/StandardKvParser_10:0', 'TensorDict/StandardKvParser_16:0', 'TensorDict/StandardKvParser_4:0', 'TensorDict/StandardKvParser_9:0']
    eged_list = pb2tensor(path1, data, list2, tag)

    feed_dict = dict(zip(list3, eged_list))
    data1 = pb2tensor(path1, data, list1, tag)
    data2 = pb2tensor(path2, data, list1, tag, feed_dict)
    return data1, data2

def check_precision(data1, data2, suffix = 0.01):
  if np.max(np.abs(data1 - data2)) > suffix:
    return False
  return True

if __name__ == "__main__":
    import sys
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    data =  gen_data(path1, bs=1)

    global tag
    tag = "serve"

    # output_op_names = ['TensorDict/StandardKvParser:0', 'TensorDict/StandardKvParser:10', 'TensorDict/StandardKvParser:16', 'TensorDict/StandardKvParser:4', 'TensorDict/StandardKvParser:9']
    result = [[],[],[]]
    oom = [[],[],[]]
    for k in range(100):
      data1, data2 =run_embedruntime(path1, path2, data[k])
    #   data1, data2 = compare_model(path1, path2, d, tag="serve")
      for i in range(len(data1)):
        assert data1[i].shape == data2[i].shape
        result[i].append(np.max(np.abs(data1[i] - data2[i])))
        if not check_precision(data1[i], data2[i]):
          oom[i].append(k)
      print(np.max(result[0]))
      print(np.max(result[1]))
      print(np.max(result[2]))
      if k % 5 == 0:
        print("vec1", len(result[0]), oom[0])
        print("vec2", len(result[1]), oom[1])
        print("vec3", len(result[2]), oom[2])


    # i = 0
    # name_list = ['TensorDict/StandardKvParser:0', 'TensorDict/StandardKvParser:10', 'TensorDict/StandardKvParse:16', 'TensorDict/StandardKvParser:4', 'TensorDict/StandardKvParser:9']
    # for d in data:
    #   i += 1
    #   data1 = run_model(path1, d, name_list, tag)
    #   np.save(str(i) + ".npy", data1)
    #   if i > 10:
    #     break

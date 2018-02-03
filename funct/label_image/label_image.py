import cv2
import numpy as np
import os
import tensorflow as tf
import time

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(file_reader, channels = 3,name='png_reader')
  elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(tf.image.decode_gif(file_reader,name='gif_reader'))
  elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
      image_reader = tf.image.decode_jpeg(file_reader, channels = 3,name='jpeg_reader')
      float_caster = tf.cast(image_reader, tf.float32)
      dims_expander = tf.expand_dims(float_caster, 0);
      resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
      normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      sess = tf.Session()
      result = sess.run(normalized)
      return result

def read_png_tensor_image_file(image_code, input_height=299, input_width=299, input_mean=0, input_std=255):
          image_reader = image_code
          # image_reader = tf.image.decode_png(image_code)
          float_caster = tf.cast(image_reader, tf.float32)
          dims_expander = tf.expand_dims(float_caster, 0);
          resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
          normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
          sess = tf.Session()
          result = sess.run(normalized)
          return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def label_image(image_code):
    graph = load_graph("./funct/label_image/label/retrained_graph.pb")
    t = read_png_tensor_image_file(image_code,input_height=224,input_width=224,input_mean=128,input_std=128)
    label_file = "./funct/label_image/label/retrained_labels.txt"
    input_name = "import/" + "input"
    output_name = "import/" + "final_result"
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
    # return results
    return(labels[results.tolist().index(max(results.tolist()))])
    # for i in top_k:
    #   print(labels[i], results[i])

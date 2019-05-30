"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
    --checkpoint=model.ckpt-399731\
    --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
import numpy as np
import PIL.Image

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session
from google.protobuf.json_format import MessageToDict

import common_flags
import datasets
import data_provider
import argparse
import predict_pb2

import math
import pdb
import cv2
import os


def get_dataset_image_size(dataset_name):
    # Ideally this info should be exposed through the dataset interface itself.
    # But currently it is not available by other means.
    ds_module = getattr(datasets, dataset_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
    return width, height

def load_images(image_dir, batch_size, dataset_name):
    max_w = 320
    new_h = 64
    files = os.listdir(image_dir)
    images_actual_data = []
    for file in files:
        im = cv2.imread(os.path.join(image_dir, file))
        h, w, d = im.shape
        unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
        if unpad_im.shape[1] > max_w:
            images_actual_data.append(cv2.resize(im, (320, 64), interpolation = cv2.INTER_AREA))
        else:
            images_actual_data.append(cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0]))
    return images_actual_data, files

def create_model(batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)
    raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, 3])
    images = tf.map_fn(data_provider.preprocess_image, raw_images,
                       dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints

def run(checkpoint, batch_size, dataset_name, image_dir, out_dir):
    images_data, files = load_images(image_dir, batch_size, dataset_name)

    for image, file in zip(images_data, files):
        chars_logit, chars_log_prob, predicted_chars, predicted_scores, predicted_text = sess.run([endpoints.chars_logit, endpoints.chars_log_prob, endpoints.predicted_chars, endpoints.predicted_scores, endpoints.predicted_text], feed_dict={images_placeholder: [image]})
        predictions = (file, chars_logit, chars_log_prob, predicted_chars, 
                            predicted_scores, predicted_text)
    return predictions


def predict_single_image(image_dir):
    word_can = {}
    prediction = run(FLAGS.checkpoint, FLAGS.batch_size, FLAGS.dataset_name, FLAGS.image_dir, FLAGS.out_dir)
    word_can["word"] = prediction[5].tolist()[0]
    char_cans = []
    for log_prob in prediction[2][0]:
        char_can = {}
        prob = []
        for i in range(len(log_prob)):
            prob.append({"char": charsets[str(i)], "prob": math.exp(log_prob[i])})
        char_can["prob"] = prob
        char_cans.append(char_can)
    word_can["char_cans"] = char_cans
    return word_can


def predict(request):
    """
    request: request = {
        "info": {
            "file": "file-path-or-image-id-for-reference"
        },
        "data": {
            "fields": {
                "name": {
                    "cut_type": "char-or-word",
                    "images": [
                        "img_word1_path",
                        "img_word2_path",
                        "..."
                    ]
                },
                "id_number": {
                    "similar to other fields": null
                },
                "birthday": {
                    "similar to other fields": null
                },
                "resident": {
                    "similar to other fields": null
                },
                "other": {
                    "similar to other fields": null
                }
            }
          }
        }
    }

    response: response = {
        "info": {
            "file": "file-path-or-image-id-for-reference"
        },
        "data": {
            "fields": {
                "name": {
                    "word_cans": [
                        {
                            "word": "thanh",
                            "char_cans": [
                                {
                                    "prob": [
                                        {"char": U, "prob": 0.4},
                                        {"char": u, "prob": 0.6}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }
    """
    # protobuf to dict
    request = MessageToDict(request)
    print(request)
    # define response
    dict_response = {
        "info": request["info"],
        "data": {
            "fields": {

            }
        }
    }
    # run model and predict candidates
    for field_key in request["data"]["fields"].keys():
        cut_type = request["data"]["fields"][field_key]["cuttype"]
        images   = request["data"]["fields"][field_key]["images"]
        dict_response["data"]["fields"][field_key] = {}
        dict_response["data"]["fields"][field_key]["word_cans"] = []
        for image in images:
            word_can = predict_single_image(image)
            dict_response["data"]["fields"][field_key]["word_cans"].append(word_can)
    response = predict_pb2.response(info=dict_response["info"], data=dict_response["data"])
    return response


FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_string('image_dir', 'images', '')
flags.DEFINE_string('out_dir', 'label.txt', '')

charsets = common_flags.get_charset()

images_placeholder, endpoints = create_model(FLAGS.batch_size, FLAGS.dataset_name)
session_creator = monitored_session.ChiefSessionCreator(
                  checkpoint_filename_with_path=FLAGS.checkpoint)
sess = monitored_session.MonitoredSession(session_creator=session_creator)
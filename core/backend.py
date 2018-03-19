from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os.path
import os
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
from ffmpy import FFmpeg
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow import saved_model as sm
import PIL.Image as Image
import numpy as np
import cv2
import logging

from config import DEFAULT_MODEL_PATH, DEFAULT_MODEL_DIR, BATCH_SIZE, CROP_SIZE, CHANNELS, NUM_FRAMES_PER_CLIP

logger = logging.getLogger()

# This methos is adapted from https://github.com/hx173149/C3D-tensorflow/blob/master/input_data.py
def get_frames_data(filename, num_frames_per_clip, seed=84):
  '''Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays'''
  ret_arr = []
  s_index = 0
  for parent, dirnames, filenames in os.walk(filename):
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    # fixed random seed for each call
    random.seed(seed)
    s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    logger.debug('Total frames: {}; start index: {}, end index: {}'.format(len(filenames), s_index, s_index + num_frames_per_clip))

    for i in range(s_index, s_index + num_frames_per_clip):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)
  return ret_arr, s_index


def convert_video_to_frames(filename):
    '''Splits the video up into 5 frames per second and saves images in directory
    :param filename: Name of the video to be processed
    :return: directory created
    '''
    video_dir = os.path.splitext(filename)[0]
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        ff = FFmpeg(inputs={filename: None},
                    outputs={video_dir + '/%05d.jpg': '-vf fps=5 -nostats -loglevel 0'})
        ff.run()
    return video_dir


def process_frames(dirname, means, batch_size=BATCH_SIZE, num_frames_per_clip=NUM_FRAMES_PER_CLIP, crop_size=CROP_SIZE):
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    img_datas = []
    data = []

    if len(tmp_data) != 0:
        for j in xrange(len(tmp_data)):
            img = Image.fromarray(tmp_data[j].astype(np.uint8))
            if img.width > img.height:
                scale = float(crop_size)/float(img.height)
                img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
            else:
                scale = float(crop_size)/float(img.width)
                img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
            crop_x = int((img.shape[0] - crop_size)/2)
            crop_y = int((img.shape[1] - crop_size)/2)
            img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :] - means[j]
            img_datas.append(img)
        data.append(img_datas)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)

    np_arr_data = np.array(data).astype(np.float32)
    return np_arr_data


class ModelWrapper(object):
    def __init__(self, path=DEFAULT_MODEL_PATH, model_dir=DEFAULT_MODEL_DIR):
        logger.info('Loading model from: {}...'.format(path))
        sess = tf.Session(graph=tf.Graph())
        # load the graph
        saved_model_path = '{}/{}'.format(path, model_dir)
        model_graph_def = sm.loader.load(sess, [sm.tag_constants.SERVING], saved_model_path)
        sig_def = signature_def_utils.get_signature_def_by_key(model_graph_def, sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        input_name = sig_def.inputs['inputs'].name
        output_name = sig_def.outputs['scores'].name

        # Load labels from file
        label_file = codecs.open('./{}/labels.txt'.format(path), "r", encoding="utf-8")
        labels = [label.strip('\n') for label in label_file.readlines()]

        self.labels = labels

        # set up instance variables and required inputs for inference
        self.sess = sess
        self.model_graph_def = model_graph_def
        self.output_tensor = sess.graph.get_tensor_by_name(output_name)
        self.input_name = input_name
        self.output_name = output_name

        self.means = np.load('./{}/crop_mean.npy'.format(path)).reshape([NUM_FRAMES_PER_CLIP, CROP_SIZE, CROP_SIZE, CHANNELS])
        logger.info('Loaded model')

    def predict(self, video_path):
        dir_name = convert_video_to_frames(video_path)
        test_images = process_frames(dir_name, self.means)
        predict_score = self.sess.run(self.output_tensor, feed_dict={self.input_name: test_images})

        # take mean of scores per frame for the average predicted video score
        predict_score = predict_score.mean(axis=0)
        # Take top 3 labels
        predict_idx = np.argsort(predict_score)[::-1][:3]
        predictions = []
        for label in predict_idx:
            predictions.append((label, self.labels[label], predict_score[label]))
        return predictions

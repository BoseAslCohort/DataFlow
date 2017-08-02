#!/usr/bin/env python

"""
Isaac Julien
"""

import os

import apache_beam as beam
import argparse

import apache_beam.options.pipeline_options
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

"""
>>>
"""

import tensorflow as tf
import numpy as np
import random


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def frame_level_data_read(tf_seq_example):
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        rgb_frame.append(np.fromstring(
            tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0], dtype=np.uint8).astype(
            float))
        audio_frame.append(np.fromstring(
            tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0], dtype=np.uint8).astype(
            float))

    return rgb_frame, audio_frame


def peturb_frames(rgb_frame, audio_frame, data_gen_param):
    frame_len = np.shape(rgb_frame)[0]
    num_frames = np.ceil(frame_len * data_gen_param['keep_ratio'])
    rgb_test = random.sample(rgb_frame, int(num_frames))
    audio_test = random.sample(audio_frame, int(num_frames))
    temp = []
    temp_audio = []
    for item in range(len(rgb_test)):
        temp.append(Dequantize(rgb_test[item], max_quantized_value=2, min_quantized_value=-2))
        temp_audio.append(Dequantize(audio_test[item], max_quantized_value=2, min_quantized_value=-2))
    video_features = np.mean(temp, axis=0)
    video_audio_features = np.mean(temp_audio, axis=0)

    return video_features, video_audio_features


def generate_examples(example, data_gen_param={'keep_ratio': 0.65, 'gen_videos': 25}):
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    vid_ids = []
    labels = []
    import time
    curr_time = time.time()

    labels = tf_seq_example.context.feature['labels'].int64_list
    vid_ids_orig = tf_seq_example.context.feature['video_id'].bytes_list.value[0]
    example_list = []

    rgb_frame, audio_frame = frame_level_data_read(tf_seq_example)
    for i in range(data_gen_param['gen_videos']):

        if i < 1:
            rgb_test, audio_test = peturb_frames(rgb_frame, audio_frame, {'keep_ratio': 1, 'gen_videos': 2})
            vid_ids = vid_ids_orig
        else:
            vid_ids = vid_ids_orig + str(i)
            rgb_test, audio_test = peturb_frames(rgb_frame, audio_frame, data_gen_param)

        example_list.append(tf.train.Example(features=tf.train.Features(feature={
            'labels':
                tf.train.Feature(int64_list=labels),
            'video_id':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[vid_ids])),
            'mean_rgb':
                tf.train.Feature(float_list=tf.train.FloatList(value=rgb_test.astype(float))),
            'mean_audio':
                tf.train.Feature(float_list=tf.train.FloatList(value=audio_test.astype(float))),
        })))

    return example_list


"""
<<<
"""


class Process(beam.DoFn):
    def process(self, element):
        import tensorflow as tf
        return generate_examples(element)


class Process2(beam.DoFn):
    def process(self, element):
        import tensorflow as tf
        tf_example = tf.train.SequenceExample.SerializeToString(element)
        return [tf_example]


if __name__ == '__main__':
    print
    "Running..."

    parser = argparse.ArgumentParser(description='DataFlow for Video-Level data augmentation')

    parser.add_argument('--inputdir', default='gs://youtube-8m/input/', help='Input directory')
    parser.add_argument('--outputdir', default='gs://youtube-8m/output/', help='Output directory')

    options, pipeline_args = parser.parse_known_args()

    input_pattern = os.path.join(options.inputdir, "*.tfrecord")
    output_prefix = os.path.join(options.outputdir, "test")

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    (p
     | 'read' >> beam.io.tfrecordio.ReadFromTFRecord(input_pattern)
     | 'process' >> beam.ParDo(Process())
     | 'process2' >> beam.ParDo(Process2())
     | 'write' >> beam.io.tfrecordio.WriteToTFRecord(output_prefix)
     )
    p.run()




#!/usr/bin/env python

"""
Isaac Julien
"""

import apache_beam as beam
import argparse
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

class Process(beam.DoFn):
   def process(self, element):
      import tensorflow as tf
      tf_example = tf.train.SequenceExample.FromString(element)
      return [tf_example]


if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='DataFlow for Video-Level data augmentation')

   parser.add_argument('--inputdir', default='gs://youtube8m-ml-us-east1/1/frame_level/train/', help='Input directory')
   parser.add_argument('--outputdir', default='gs://processedframeleveloutput/outputs/', help='Output directory')

   options, pipeline_args = parser.parse_known_args()

   input_pattern = '{0}*tfrecord'.format(options.inputdir)

   output_prefix = options.outputdir + "test"

   p = beam.Pipeline(argv=pipeline_args)


   pipeline_options = PipelineOptions(pipeline_args)
   pipeline_options.view_as(SetupOptions).save_main_session = True
   p = beam.Pipeline(options=pipeline_options)

   (p
   | 'read' >> beam.io.tfrecordio.ReadFromTFRecord(input_pattern)
   | 'process' >> beam.ParDo(Process())
   | 'write' >> beam.io.tfrecordio.WriteToTFRecord(output_prefix)
   )
   p.run()
#!/usr/bin/env python

"""
Isaac Julien
"""

import apache_beam as beam

#import org.apache.beam.sdk.io.TFRecordIO as tfrecordio
import apache_beam.io.tfrecordio as tfrecordio

import argparse

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='DataFlow for Video-Level data augmentation')

   parser.add_argument('--inputdir', default='gs://youtube8m-ml-us-east1/1/frame_level/train/', help='Input directory')
   parser.add_argument('--outputdir', default='gs://processedframeleveloutput/outputs/', help='Output directory')

   options, pipeline_args = parser.parse_known_args()

   p = beam.Pipeline(argv=pipeline_args)

   input = '{0}*tfrecord'.format(options.inputdir)

   output_prefix = options.outputdir + "test"


   (p
      | 'read' >> tfrecordio.ReadFromTFRecord("gs://testinput/input/*")
      | 'write' >> tfrecordio.WriteToTFRecord(output_prefix)
   )

   p.run()
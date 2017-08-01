#!/usr/bin/env python

"""
Isaac Julien
"""

import apache_beam as beam
import apache_beam.io.gcp.gcsfilesystem.GCSFileSystem as gcsio

import argparse

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='')

   parser.add_argument('--inputdir', default='gs://youtube8m-ml-us-east1/1/frame_level/train/', help='Input directory')
   parser.add_argument('--outputdir', default='gs://processedframeleveloutput/outputs/', help='Output directory')

   options, pipeline_args = parser.parse_known_args()

   p = beam.Pipeline(argv=pipeline_args)

   input = '{0}*.tfrecord'.format(options.input)
   output_prefix = options.outputdir + "test"

   (p
      | 'GetFilenames' >> gcsio.match(input)
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run()
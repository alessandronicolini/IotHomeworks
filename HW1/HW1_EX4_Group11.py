import numpy
import pandas as pd

import os
import argparse

import datetime
from datetime import timezone

import wave

import tensorflow as tf

#arguments

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="input directory")
parser.add_argument("--output",  type=str, help="output file")
args = parser.parse_args()

path = args.input
csv_path = str(path)+'/samples.csv'
out_path = args.output

#read csv as pandas dataframe

df = pd.read_csv(csv_path, usecols=[0, 1, 2, 3, 4], names=['date', 'time', 'temp', 'hum', 'audio'])

#create tfrecord dataset

with tf.io.TFRecordWriter(out_path) as writer:
            for date, time, filename, temp, hum in zip(df.date, df.time, df.audio, df.temp, df.hum):

                #convert time and date into posix timestamp
                year = int(date.split('/')[2])
                month = int(date.split('/')[1])
                day = int(date.split('/')[0])
                second = int(time.split(':')[2])
                minute = int(time.split(':')[1])
                hour = int(time.split(':')[0])

                dt = datetime.datetime(year, month, day, hour, minute, second, 0, tzinfo=timezone.utc)
                timestamp = int(dt.timestamp())

                #open audio file and save its content as bytes object
                file_path = str(path) + '/' + str(filename)

                fp = wave.open(file_path)
                nchan = fp.getnchannels()
                N = fp.getnframes()
                audio_data = fp.readframes(N * nchan)

                #create features and the dictionary, then store them in the tfrecord dataset
                datetime_feature = tf.train.Feature(int64_list = tf.train.Int64List(value=[timestamp]))
                temp_feature = tf.train.Feature(int64_list = tf.train.Int64List(value=[temp]))
                hum_feature = tf.train.Feature(int64_list = tf.train.Int64List(value=[hum]))
                audio_feature = tf.train.Feature(bytes_list = tf.train.BytesList(value=[audio_data]))

                mapping = {
                    'datetime' : datetime_feature,
                    'temperature' : temp_feature,
                    'humidity' : hum_feature,
                    'audio' : audio_feature
                }

                feature_map = tf.train.Features(feature=mapping)

                example = tf.train.Example(features=feature_map)

                writer.write(example.SerializeToString())

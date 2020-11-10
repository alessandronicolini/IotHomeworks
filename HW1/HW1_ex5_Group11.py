import argparse
import numpy as np
import pyaudio
import time
from io import BytesIO
import  subprocess


def main():

	# read input
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_samples", type=int, help="number of samples to be recorded")
	parser.add_argument("--output",  type=str, help="directory where results are stored")
	args = parser.parse_args()

	# define useful variavles
	chunk = 1024 # length of a chunk
	channels = 1
	rate = 48000
	bytes_per_chunk = 2*chunk # each value is int16, 2 bytes
	num_chunks = int(rate/chunk) # total number of chunks

	# make a pyaudio instance
	pa = pyaudio.PyAudio()

	# set freq to min value
	subprocess.Popen([
		'sudo',
		'sh',
		'-c',
		"echo powersave > /sxys/devices/system/cpu/cpufreq/policy0/scaling_governor"
	])

	# initilaize a data container array
	frame = BytesIO(bytes(bytes_per_chunk*num_chunks))
	view = frame.getbuffer()

	# cycle for each sample
	for sample in range(args.num_samples):

		start = time.time()

		# open stream object as input
		stream = pa.open(
    		format=pyaudio.paInt16,
   			channels=channels,
    		rate=rate,
    		input=True,
    		frames_per_buffer=chunk
    		)

		# create audio frame
		for i in range(num_chunks):
			view[i*bytes_per_chunk:(i+1)*bytes_per_chunk] = stream.read(chunk)

		# stop and close stream
		stream.stop_stream()
		stream.close()

		# set max freq
		subprocess.Popen([
			'sudo',
			'sh',
			'-c',
			"echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
        ])

		# set low freq
		subprocess.Popen([
			'sudo',
			'sh',
			'-c',
			"echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
        ])

		# compute cycle time
		end = time.time() - start
		print(end)

	pa.terminate()

if __name__ == '__main__':
	main()

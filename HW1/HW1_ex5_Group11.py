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

	# reset monitor
	subprocess.call([
		'sudo',
		'sh',
		'-c',
		"echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"
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

		end_open_stream = time.time()

		# create audio frame
		start_frame = time.time()

		for i in range(num_chunks):
			view[i*bytes_per_chunk:(i+1)*bytes_per_chunk] = stream.read(chunk)
			if time.time()-start_frame > 0.93:
				popen_start = time.time()
				subprocess.Popen([
					'sudo',
					'sh',
					'-c',
					"echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
				])
				popen_end = time.time()
		audio = np.frombuffer(frame.getvalue(), dtype = np.int16)
		end_frame = time.time()

		# stop and close stream
		start_close_stream = time.time()

		stream.stop_stream()
		stream.close()

		end_close_stream = time.time()

		subprocess.Popen([
			'sudo',
			'sh',
			'-c',
			"echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
		])

		# cycle time
		end = time.time()

		print("cycle time %s"%(end-start))
		print("open stream time %s"%(end_open_stream - start))
		print("popen fmax time %s"%(popen_end-popen_start))
		print("closing stream time %s"%(end_close_stream - start_close_stream))
		print("frame+close_stream time %s"%(end_close_stream - start_frame))

	pa.terminate()

if __name__ == '__main__':
	main()

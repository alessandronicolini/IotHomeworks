import argparse
import numpy as np
import pyaudio
import time
from scipy import signal
from io import BytesIO
import subprocess
import math
import tensorflow as tf
import os




# read input
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, help="number of samples to be recorded")
parser.add_argument("--output",  type=str, help="directory where results are stored")
args = parser.parse_args()

# make output folder
try:
	os.mkdir(args.output)
except FileExistsError:
	pass

# define useful variavles
chunk_size = 4800  # length of a chunk
channels = 1
rate = 48000
bytes_per_chunk = 2*chunk_size # each value is int16, 2 bytes
num_chunks = int(rate/chunk_size) # total number of chunks

# make a pyaudio instance
pa = pyaudio.PyAudio()

# make stream instance
stream = pa.open(
    format=pyaudio.paInt16,
   	channels=channels,
    rate=rate,
    input=True,
    frames_per_buffer=chunk_size,
	start=False
)

# initilaize a data container array
frame = BytesIO(bytes(bytes_per_chunk*num_chunks))
view = frame.getbuffer()

# initilaize cycle time list
cycle_time = []

# mel weight matrix
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
	num_mel_bins=40,
	num_spectrogram_bins=321,
	sample_rate=16000,
	lower_edge_hertz=20,
	upper_edge_hertz=4000
)

# reset monitor
subprocess.call([
	'sudo',
	'sh',
	'-c',
	"echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"
])

# cycle for each sample
for sample in range(args.num_samples):

	start = time.time()

	# start audio stream
	stream.start_stream()

	for i in range(num_chunks):
		if i== 0:
			subprocess.Popen([
				'sudo',
				'sh',
				'-c',
				"echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
			])
		elif i == 9:
			subprocess.Popen([
				'sudo',
				'sh',
				'-c',
				"echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
			])
		view[i*bytes_per_chunk:(i+1)*bytes_per_chunk] = stream.read(chunk_size)

	# stop and close stream
	stream.stop_stream()

	# preprocessing
	audio = np.frombuffer(frame.getvalue(), dtype=np.int16)
	audio = signal.resample_poly(audio, 1,3)
	tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)
	stft = tf.signal.stft(tf_audio, frame_length=640, frame_step=320, fft_length=640) # length is freq(16KHz)*time(40/20 ms)
	spectrogram = tf.abs(stft)
	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
	mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]
	conversion = tf.io.serialize_tensor(mfccs)
	tf.io.write_file(args.output+"/mfccs"+str(sample)+".bin", conversion)

	# cycle time
	cycle_time.append(time.time()-start)

# close stream and perminate pa
stream.close()
pa.terminate()

# print times
for cycle_t in cycle_time:
	print(cycle_t)

# check times
subprocess.call([
	'cat',
	"/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"
])

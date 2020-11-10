import argparse
import pyaudio
import time
from io import BytesIO


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_samples", type=int, help="number of samples to be recorded")
	parser.add_argument("--output",  type=str, help="directory where results are stored")
	args = parser.parse_args()

	pa = pyaudio.PyAudio()

	for i in range(args.num_samples):
		start = time.time()

		# open stream object as input
		stream = pa.open(
    		format=pyaudio.paInt16,
   			channels=1,
    		rate=48000,
    		input=True,
    		frames_per_buffer=1024
    		)

		frames = []


		for i in range(int(48000/1024)):
			data = stream.read(1024)
			frames.append(data)

		stream.stop_stream()
		stream.close()
		end = time.time() - start
		print(end)

	pa.terminate()

if __name__ == '__main__':
	main()

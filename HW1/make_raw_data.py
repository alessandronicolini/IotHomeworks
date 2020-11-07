import csv
from sys import argv

# number of csv rows
rows = 10

audio_files = ["audio"+str(i)+".wav" for i in range(1,5)]
date = "18/10/2020"

with open("raw_data/samples.csv", 'w', newline='') as file:
	writer = csv.writer(file)
	for min, sec, i in zip(range(40,40+rows+1), range(10,10+ (rows+1)*2, 2), range(1, rows+1)):
		writer.writerow([date, "9:"+str(min)+":"+str(sec), "21", "65", audio_files[i%4]])




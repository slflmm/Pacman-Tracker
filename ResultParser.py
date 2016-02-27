import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

def main():

	tracked = 0
	not_tracked = 0
	total_frames = 0
	avg_error = 0
	N_error = 0
	min_error = np.inf
	max_error = 0
	error_values = []
	episode_lengths = []

	lines = [line.strip() for line in open('trackingResults.txt')]
	for l in lines:
		tokens = l.split(' ')
		if tokens[1]=='SPLIT':
			 if tokens[3]=='not':
			 	not_tracked+=1
			 if tokens[3]=='tracked':
			 	tracked+=1
		if tokens[0]=='BLOB':
			total_frames += int(tokens[6])
			error = float(tokens[9])
			avg_error += error
			if error < min_error:
				min_error = error
			if error > max_error:
				max_error = error
			N_error += 1
			error_values.append(error)
			episode_lengths.append(int(tokens[6]))

	mean_particle = tracked*1.0/(tracked+not_tracked)
	ordered_error_values = np.sort(error_values)
	median = error_values[len(ordered_error_values)/2]

	plt.hist(error_values, 20)
	plt.xlabel('Average error')
	plt.ylabel('Count')
	plt.title('Error frequencies')
	plt.savefig('histogram.png')
	plt.clf()

	plt.plot(episode_lengths, error_values, 'ro')
	plt.xlabel('Episode length')
	plt.ylabel('Average error')
	plt.title('Length of tracking episode vs its average error')
	plt.savefig('scatter.png')

	correlation= pearsonr(episode_lengths, error_values)

	print "With blobs:"
	print "    Average tracking duration: " + str(total_frames/N_error)
	print "    Average average error: " + str(avg_error/N_error)
	print "    Min average error: " + str(min_error)
	print "    Max average error: " + str(max_error)
	print "    Median average error: " + str(median)
	print "    Correlation between length and errors: " + str(correlation)
	print "During splits:"
	print "    Tracked: " + str(tracked)
	print "    Not tracked: " + str(not_tracked)
	print "    Success: " + str(mean_particle)

if __name__ == "__main__": main()

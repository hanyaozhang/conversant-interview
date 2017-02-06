'''
Hanyao Zhang, 2/3/2017 for Conversant, LLC

DESCRIPTION
Takes in a data file in the same format and encoding as the included 'data.Montoya.txt'
Attempts to provide meaningful insights into the data given.

USAGE INSTRUCTIONS
Please place the file to run analytics on in the same folder as this script.
Then run from the command line using: "python analyze.py [FILENAME]"
For our purposes, this should be: "python analyze.py data.Montoya.txt"

Written with Python 2.7.3, and requires that *NumPy* and *matplotlib* be installed.

Assumes all input is valid, though not necessarily in order.
Assumes all entries are rtb.requests

Note: I realized very late in the process of coding this that my decision to store entries
as a tuple in the data_center object was somewhat careless, and results in quite a bit of
added runtime. Instead, I should have saved values and times in separate arrays and zipped
them for sorting.
'''

import sys
from operator import itemgetter
import matplotlib.pyplot as plt
import math
import numpy
import heapq
import os

#CHANGE THESE AS NEEDED
GAPS_PERCENTAGE = 0.01 #percentage of largest time intervals to report
DERIVS_PERCENTAGE = 0.01 #percentage of d_value/d_time to report
NUM_FILTERING_PASSES = 3 #<INT> number of filtering passes to make
FILTERING_THRESH = 3 #<INT> each pass will filter out anything +/- FILTERING_THRESH*SD
GRAPH_WIDTH_INCHES = 18 #width of all graphs, in inches
GRAPH_HEIGHT_INCHES = 9 #height of all graphs, in inches

#keep track of all data centers we've encountered in the data
known_data_centers = {}


#-------------------------------------------------------------------------
#DATA_CENTER CLASS
#-------------------------------------------------------------------------
#handle data for each data center separately; can be used for all
class data_center:
	def __init__(self, name):
		self.name = name
		self.entries = [] #will store entries (time, value, name) as a tuple
		self.intervals = [] #inelegant, but should be used ONLY after analyze_times
		self.value_median = None
		self.value_std_dev = None

	'''
	Add a data entry to the object.
	'''
	def add_entry(self, time, value, name):
		if name == "do not save": #for dealing with individual data centers
			self.entries.append((time, value))
		else:
			self.entries.append((time, value, name))

	'''
	Sorts the self.entries array by the first item in each entry tuple: the time.
	'''
	def sort_by_time(self):
		self.entries.sort(key=itemgetter(0))

	'''
	Goes through the list and sums up value entries from the same time.
	Should only be needed for the 'ALL' data center
	'''
	def sanitize(self):
		mark_for_delete = []
		#the entry for subsequent entries with matching times to be summed into
		curr_idx = 0
		for idx, entry in enumerate(self.entries[1:]):
			#matching times
			if entry[0] == self.entries[curr_idx][0]:
				#another reason not to use tuples
				self.entries[curr_idx] = (self.entries[curr_idx][0],\
					self.entries[curr_idx][1]+entry[1], self.entries[curr_idx][2])
				mark_for_delete.append(idx)
			else:
				curr_idx = idx

		#deleting items while iterating; must be done starting from the end
		for idx in reversed(mark_for_delete):
			del self.entries[idx]

	'''
	Returns a tuple containing the total range of time, as well as min, max, and average
	intervals between entries.
	'''
	def analyze_times(self):
		#range of time is last recorded time minus the first
		time_range = self.entries[-1][0] - self.entries[0][0]

		#these variables help us figure out if entries are spaced evenly timewise
		equal_intervals = True
		temp_diff = self.entries[1][0] - self.entries[0][0]

		#calculate all time intervals
		prev_entry = self.entries[0]
		for entry in self.entries[1:]:
			time_diff = entry[0] - prev_entry[0]
			self.intervals.append((time_diff, prev_entry[0])) #match interval to starting time

			#not equal to the last interval; no point in setting multiple times
			if (time_diff != temp_diff) and (equal_intervals == True):
				equal_intervals = False

			prev_entry = entry

		biggest_gaps = []
		#not equal, so we should find the biggest gaps
		if equal_intervals == False:
			#store the given percentage (rounding up) largest gaps
			gaps_to_report = int(math.ceil(len(self.intervals)*GAPS_PERCENTAGE))
			biggest_gaps = heapq.nlargest(gaps_to_report, self.intervals, key=itemgetter(0))

		#pull out the intervals ONLY for use with numpy
		time_intervals = list(map(lambda x: x[0], self.intervals))

		#are useless if all intervals are equal; will be ignored per equal_intervals
		#don't think median is particularly useful here
		hi = max(time_intervals)
		low = min(time_intervals)
		mean = numpy.mean(time_intervals)
		std_dev = numpy.std(time_intervals)

		interval_stats = (hi, low, mean, std_dev)

		return (time_range, equal_intervals, biggest_gaps, interval_stats)
		
	'''
	Returns a tuple containing the max, min, mean, and median values, as well as standard
	deviation [and other meaningful statistics]
	'''
	def analyze_values(self):
		#we could use itemgetter, but numpy functions don't have a "key" argument
		#as a result, we'll need to pull out the values into a separate list
		values = list(map(lambda x: x[1], self.entries))

		total = sum(values)
		hi = max(values)
		low = min(values)
		mean = numpy.mean(values)
		self.value_median = numpy.median(values)
		self.value_std_dev = numpy.std(values)

		#note that we return max, min, etc. with the entire entry
		return (hi, low, mean, self.value_median, self.value_std_dev, total)

	'''
	Creates an array of finite difference rate-of-change values. Returns this, the mean,
	std_dev, and the largest changes
	'''
	def analyze_ROC(self):
		#calculate the rates of change
		derivatives = []
		numpy_derivs = []
		prev_entry = None
		for entry in self.entries:
			if prev_entry == None:
				prev_entry = entry
				continue
			d_time = entry[0] - prev_entry[0]
			d_value = entry[1] - prev_entry[1]
			if d_time != 0:
				derivatives.append(((d_value/d_time), prev_entry[0])) #match change to starting time
				numpy_derivs.append(d_value/d_time)
			prev_entry = entry

		#get the largest changes
		derivs_to_report = int(math.ceil(len(derivatives)*DERIVS_PERCENTAGE))
		largest_changes = heapq.nlargest(derivs_to_report, derivatives, key=itemgetter(0))

		mean = numpy.mean(numpy_derivs)
		std_dev = numpy.std(numpy_derivs)

		return (derivatives, largest_changes, mean, std_dev)

	'''
	Removes any outliers.
	Again, sloppy design: but please ensure analyze_values is called before this function.
	'''
	def kill_outliers(self):
		#define an outlier as something +/- 2 standard deviations beyond the median
		outlier_thresh = FILTERING_THRESH*self.value_std_dev

		self.entries = filter(lambda x: (x[1] <= (self.value_median + outlier_thresh)) and\
			(x[1] >= (self.value_median - outlier_thresh)), self.entries)

	'''
	Print out all data entries
	'''
	def print_entries(self):
		for entry in self.entries:
			print entry

#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#MAIN FUNCTIONS
#-------------------------------------------------------------------------

'''
Takes in the name of the file to be parsed.
Returns a data_center object containing all the parsed entries.
'''
def parse_file(filename):
	try:
		#create a data_center object, named ALL
		all_centers = data_center('ALL')

		#open file for reading, and parse the contents line by line
		with open(filename) as f:
			for line in f:
				#print line #debugging, comment out later
				#discard the line if it is not an rtb request
				if line.split()[0] != "rtb.requests":
					continue

				#use split to split line into individual "words" and numbers
				words = line.split()
				parsed_time = int(words[1]) #time always represented as an int
				parsed_value = float(words[2]) #time always represented as float
				parsed_name = words[3].split("=")[1]

				#add the data to the "ALL" data_center object
				all_centers.add_entry(parsed_time, parsed_value, parsed_name)

		#all_centers.print_entries() #check that all data was extracted correctly

		#sort entries in the "ALL" data_center by time
		all_centers.sort_by_time()

		return all_centers

	except Exception, e:
		#an error arose while parsing the file
		print "[*] Unable to parse file. Error: %s", e
		sys.exit

'''
Takes in a data_center object. Calls analyze_values and prints the results.
'''
def print_value_analysis(center):
	val_res = center.analyze_values()
	print "Values Statistics\n----------------\nTotal requests: %f\nMax: %f\nMin: %f\nMean: %f\n"\
		"Median: %f\nSD: %f\n"\
		% (val_res[5], val_res[0], val_res[1], val_res[2], val_res[3], val_res[4])

'''
Takes in a data_center_object. Calls analyze_times and prints the results.
'''
def print_time_analysis(center):
	time_res = center.analyze_times()
	print "Times Statistics\n----------------\nRange: %d\n"\
		"Equal time intervals? (T/F): %r" % (time_res[0], time_res[1])
	#if not all intervals were NOT equal
	if time_res[1] == False:
		print "Max interval: %f\nMin interval: %f\nMean interval: %f\n"\
			"Interval SD: %f\n" % (time_res[3][0], time_res[3][1], time_res[3][2],\
				time_res[3][3])
		print "The largest interval(s) was/were:"
		for gap in time_res[2]:
			print "%f from time %f" % (gap[0], gap[1])
		print "\n"

'''
Takes in a data_center object. Calls analyze_ROC and prints the results.
Returns the array containing all the calculated rates of change.
'''
def print_ROC_analysis(center):
	ROC_res = center.analyze_ROC()
	print "ROC Statistics\n--------------\n"
	print "Mean ROC: %f\nROC SD: %f\n" % (ROC_res[2], ROC_res[3])
	print "The biggest change(s) was/were:"
	for change in ROC_res[1]:
		print "%f from time %f" % (change[0], change[1])
	print "\n"

	return ROC_res[0]

'''
Takes in a data_center_object, and graphs the values over time.
This assumes that the object's entries have already been sorted, and that analyze_values
has been run.

filtered - whether or not outliers have been filtered; affects where the graph will be saved
pass_num - the number of the filtering pass we're currently on
'''
def graph_values(center, filtered, pass_num):
	#gets the time and value arrays
	x = list(map(lambda x: x[0], center.entries))
	y = list(map(lambda x: x[1], center.entries))
	fig = plt.figure()
	fig.set_size_inches(GRAPH_WIDTH_INCHES, GRAPH_HEIGHT_INCHES)
	plt.bar(x, y)
	plt.title('Values for DC=%s' % (center.name))
	plt.xlabel('Time')
	plt.ylabel('Value')

	if (filtered) == True and (not os.path.exists("value_graphs/filtered_pass_%d" % (pass_num))):
		os.makedirs("value_graphs/filtered_pass_%d" % (pass_num))
	if not os.path.exists("value_graphs/unfiltered"):
		os.makedirs("value_graphs/unfiltered")

	if filtered:
		plt.savefig('value_graphs/filtered_pass_%d/%s_values.svg' % (pass_num, center.name))
	else:
		plt.savefig('value_graphs/unfiltered/%s_values.svg' % (center.name))
	plt.close(fig)
  
'''
Graphs d_value/d_time over time
'''
def graph_ROC(center, derivs):
	x = list(map(lambda x: x[1], derivs))
	y = list(map(lambda x: x[0], derivs))
	fig = plt.figure()
	fig.set_size_inches(GRAPH_WIDTH_INCHES, GRAPH_HEIGHT_INCHES)
	plt.bar(x, y)
	plt.title('ROCs for DC=%s' % (center.name))
	plt.xlabel('Time')
	plt.ylabel('Value ROC')

	if not os.path.exists("ROC_graphs"):
		os.makedirs("ROC_graphs")

	plt.savefig('ROC_graphs/%s_ROCs.svg' % (center.name))
	plt.close(fig)

'''
Calls analyze functions using the "ALL" data center; graphs values as well
''' 
def analyze_combined_center(filtered, pass_num):
	print "[*] Analyzing all data...\n"
	print "Total Entries: %d" % (len(all_centers.entries))
	print_value_analysis(all_centers)
	print_time_analysis(all_centers)
	graph_values(all_centers, filtered, pass_num)

'''
Calls analyze functions for all individual data centers; graphs values as well
'''
def analyze_individual_centers(filtered, pass_num):
	print "[*] Analyzing individual data centers...\n"
	#run analytics on each individual data center (again, with entries sorted by time)
	for key in known_data_centers:
		print "DATA CENTER: %s\n" % (known_data_centers[key].name)
		print "Total Entries: %d" % (len(known_data_centers[key].entries))
		print_value_analysis(known_data_centers[key])
		print_time_analysis(known_data_centers[key])
		graph_values(known_data_centers[key], filtered, pass_num)

#-------------------------------------------------------------------------


#grab filename from command line
file = sys.argv[1]
print "[*] Using %s as data source...\n" % (file)
print "[*] Current percentage of largest intervals to report: %s\n" % str(GAPS_PERCENTAGE)

#parse the file
all_centers = parse_file(file)

#create separate objects for each individual data center; should be sorted
for entry in all_centers.entries:
	entry_name = entry[2]
	#print entry_name
	if entry_name not in known_data_centers:
		#add an data center in the dictionary
		known_data_centers[entry_name] = data_center(entry_name)
	known_data_centers[entry_name].add_entry(entry[0], entry[1], "do not save")

#run sanitize on all_centers to combine entries with the same time
all_centers.sanitize()

#run analytics on all aggregated data (sorted by time)
analyze_combined_center(False, None)
#on individual centers
analyze_individual_centers(False, None)

#do it all again a specified number of times, after killing outliers
for i in range(NUM_FILTERING_PASSES):
	all_centers.kill_outliers()
	for key in known_data_centers:
		known_data_centers[key].kill_outliers()
	print "[*] Rerunning with outliers (outside %d*SD) removed (Pass %d).\n"\
		% (FILTERING_THRESH, i+1)
	analyze_combined_center(True, i+1)
	analyze_individual_centers(True, i+1)

#calculate derivatives and graph; do on final, FILTERED results only
#want to avoid something too complex, so sticking to finite difference
print "[*] Rate of change analysis...\n"

derivs_ret = all_centers.analyze_ROC()
print "[*] Analyzing ROCs for all data...\n"
derivs = print_ROC_analysis(all_centers)
graph_ROC(all_centers, derivs)

print "[*] Analyzing ROCs for individual data centers...\n"
for key in known_data_centers:
	print "DATA CENTER: %s\n" % (known_data_centers[key].name)
	derivs_ret = known_data_centers[key].analyze_ROC()
	derivs = print_ROC_analysis(known_data_centers[key])
	graph_ROC(known_data_centers[key], derivs)

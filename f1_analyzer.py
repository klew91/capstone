import csv
from f1_score import f1_score

topicSets = []

with open('sortedsurveytopics.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    i = 0
    topicSet = []
    for row in csvReader:
    	topicName = row[0].lower()
    	topicCluster = [word.lower() for word in row[1].split(', ')]
    	topicSet.append([topicName, topicCluster])
    	i += 1
    	if i is 10:
    		topicSets.append(topicSet)
    		topicSet = []
    		i = 0

humanSets = topicSets[:4]
machineSets = topicSets[4:]

thresholds = []

for i in range(10):

	i_sum = 0
	for index, rootSet in enumerate(humanSets):

		root_sum = 0
		for childSet in humanSets[:index] + humanSets[(index+1):]:

			f1_sum = 0
			for topic1 in rootSet[:i]:
				highest_f1 = 0
				for topic2 in childSet[:i]:

					current_f1 = f1_score(topic1[1], topic2[1])

					if current_f1 > highest_f1:
						highest_f1 = current_f1

				f1_sum += highest_f1

			f1_average = f1_sum/(i+1)
			root_sum += f1_average

		root_average = root_sum/4.00
		i_sum += root_average

	i_average = i_sum/4.00
	thresholds.append(i_average)

results = []

for i in range(10):

	i_sum = 0
	for index, rootSet in enumerate(machineSets):

		root_sum = 0
		for childSet in humanSets:

			f1_sum = 0
			for topic1 in rootSet[:i]:
				highest_f1 = thresholds[i]
				for topic2 in childSet[:i]:

					current_f1 = f1_score(topic1[1], topic2[1])

					if current_f1 > highest_f1:
						highest_f1 = current_f1

				f1_sum += highest_f1

			f1_average = f1_sum/(i+1)
			root_sum += f1_average

		root_average = root_sum/4.00
		resultStr = " ".join(["Algorithm ", str(index+1), " average F1 Score for ", str(i+1), " topics: ", str(root_average)])
		results.append(resultStr)

		diff = root_average - thresholds[i]
		percent_improved = diff
		if diff > 0:
			percent_improved = (diff/thresholds[i]) * 100

		resultStr = " ".join(["Algorithm ", str(index+1), " % improvement over humans for ", str(i+1), " topics: ", str(percent_improved), "\n"])
		results.append(resultStr)
		i_sum += root_average

	i_average = i_sum/4.00
	resultStr = " ".join(["Average F1 Score across all algorithms for ", str(i+1), " topics: ", str(i_average), "\n"])
	results.append(resultStr)

outfile = open('results.txt', 'w')

for result in results:
  outfile.write("%s\n" % result)






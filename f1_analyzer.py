import csv
from f1_score import f1_score
from orderedset import OrderedSet

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
pairings = OrderedSet([])

root_highest_f1 = 0
bestHumanIdx = 0
for i in range(10):

	i_sum = 0
	for index, rootSet in enumerate(humanSets):

		root_sum = 0
		for humanIndex, childSet in enumerate(humanSets[:index] + humanSets[(index+1):]):

			f1_sum = 0
			for topic1 in rootSet[:i+1]:
				highest_f1 = 0
				for topic2 in childSet[:i+1]:

					current_f1 = f1_score(topic1[1], topic2[1])

					if current_f1 > highest_f1:
						highest_f1 = current_f1

					secondTopicIdx = humanIndex
					if index is 0:
						secondTopicIdx += 1
					elif index is 1 and secondTopicIdx > 0:
						secondTopicIdx += 1
					elif index is 2 and secondTopicIdx is 2:
						secondTopicIdx += 1

					if current_f1 > 0.3:
						resultStr = " ".join(["Human ", str(index+1), " topic: ", topic1[0], " paired greater than 0.3 with Human ", str(secondTopicIdx+1), " topic: ", topic2[0] ])
						pairings.add(resultStr)

				f1_sum += highest_f1

			f1_average = f1_sum/(i+1)
			root_sum += f1_average

		root_average = root_sum/4.00
		i_sum += root_average

		if root_average > root_highest_f1:
			root_highest_f1 = root_average
			bestHumanIdx = index

	i_average = i_sum/4.00
	thresholds.append(i_average)

print(bestHumanIdx)

results = []
bestTopics = []
commonhuman = dict()

for i in range(10):

	i_sum = 0
	for index, rootSet in enumerate(machineSets):

		root_sum = 0
		root_highest_f1 = 0
		bestTopic = ""
		for humanIndex, childSet in enumerate(humanSets):

			f1_sum = 0
			for topic1 in rootSet[:i+1]:
				highest_f1 = thresholds[i]
				humantopic = ""
				for topic2 in childSet[:i+1]:

					current_f1 = f1_score(topic1[1], topic2[1])

					if current_f1 > highest_f1:
						highest_f1 = current_f1
						humantopic = topic2[0]

					if current_f1 > root_highest_f1:
						root_highest_f1 = current_f1
						bestTopic = topic1[0]


					if current_f1 > 0.3:
						resultStr = " ".join(["Algorithm ", str(index+1), " topic: ", topic1[0], " paired greater than 0.3 with Human ", str(humanIndex+1), " topic: ", topic2[0] ])
						pairings.add(resultStr)

				if humantopic != "" and humantopic in commonhuman:
					commonhuman[humantopic] += 1
				else:
					commonhuman[humantopic] = 1
				f1_sum += highest_f1
				#resultStr = " ".join(["Algorithm ", str(index+1), " topic: ", topic1[0], " had an average f1 score of ", str(highest_f1) ])
				#results.append(resultStr)

			f1_average = f1_sum/(i+1)
			root_sum += f1_average

		if bestTopic != "":
			resultStr = " ".join(["Best topic for Algorithm ", str(index+1), " with ", str(i+1), " topics: ", bestTopic])
			bestTopics.append(resultStr)


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

outfile = open('results.txt', 'w')

for result in results:
  outfile.write("%s\n" % result)

for pairing in pairings:
  outfile.write("%s\n" % pairing)

outfile.write("\n")

for topic in bestTopics:
  outfile.write("%s\n" % topic)

outfile.write("\n")

for topic in commonhuman:
	if commonhuman[topic] > 5:
  		outfile.write("%s\n" % " ".join([topic, " paired more than 5 times with algorithm topics"]))






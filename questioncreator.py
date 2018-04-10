import csv
from f1_score import f1_score

topicSets = []

with open('surveytopics.csv') as csvDataFile:
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

baseSet = topicSets[0][:5]
restTopicSets = topicSets[1:]
questions = []
blacklist = []

for parent_topic in baseSet:
	question = []
	question.append(parent_topic)
	parent_cluster = parent_topic[1]

	for topicSetIndex, topicSet in enumerate(restTopicSets):
		highest_f1 = 0
		index_highest = 0

		for topicIndex,topic in enumerate(topicSet):

			topicName = topic[0]
			topicCluster = topic[1]
			current_f1 = f1_score(parent_cluster, topicCluster)

			if(current_f1 > highest_f1) and ((topicSetIndex, topicIndex) not in blacklist):
				highest_f1 = current_f1
				index_highest = topicIndex

		topicHighest = topicSet[index_highest]
		question.append(topicHighest)
		blacklist.append((topicSetIndex, index_highest))

	questions.append(question)


outfile = open('surveyquestions.txt', 'w')

for i, question in enumerate(questions):

	outfile.write("Question ");
	outfile.write("%s" % i);
	outfile.write(": \n");

	for topic in question:
		outfile.write("Topic Name: ");
		outfile.write("%s\n" % topic[0]);
		outfile.write("Topic Cluster: ");
		outfile.write("%s\n\n" % ", ".join(topic[1]));







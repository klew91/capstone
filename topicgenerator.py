from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
import sys
from operator import add
from pyspark import SparkContext
import re
from wordfreq import word_frequency
import collections
from gensim.models import Word2Vec
import csv
import nltk
from nltk.collocations import *

outfile = open('data.txt', 'w')

with open('satisfaction.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        outfile.write("%s\n" % row[2])

tokenizer = RegexpTokenizer(r'\w+')

def read_words(words_file):
    return [word.lower() for line in open(words_file, 'r') for word in tokenizer.tokenize(line)]

words = read_words('data.txt')

stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if not w in stop_words]
filtered_words_list = list(filtered_words)

sc = SparkContext(appName="KeywordCluster")
filtered_words = sc.parallelize(filtered_words)

counts = filtered_words.map(lambda x: (x, 1)) \
		.reduceByKey(add) \
		.sortBy(lambda x: x[1], ascending=False)

def calc_SIP(wordTuple):
	word = wordTuple[0]
	freq = word_frequency(word, 'en')
	if freq == 0:
		freq = float(.00001)
	SIPscore = wordTuple[1]/freq
	return (word, SIPscore)

SIPscores = counts.map(calc_SIP)\
				  .filter(lambda x: x[1] > 20000)\
				  .sortBy(lambda x: x[1], ascending=False)

#SIPscores.saveAsTextFile(sys.argv[1])

sentences = [words]
model = Word2Vec(sentences, size=300, window=2, min_count=0, workers=4)

SIPscoreslist = SIPscores.collect()
clusters = []
blacklist = {}
for i, parent_word in enumerate(SIPscoreslist):
	if parent_word[0] not in blacklist:
		cluster = []
		blacklist[parent_word[0]] = 1
		cluster.append(parent_word[0])
		for word in SIPscoreslist[:i] + SIPscoreslist[(i+1):]:
			if word[0] not in blacklist:
				if model.wv.similarity(parent_word[0], word[0]) > .80:
					blacklist[word[0]] = 1
					cluster.append(word[0])
					if len(cluster) is 20:
						break
		cluster = pos_tag(cluster)
		clusters.append(cluster)

posClusters = []
for cluster in clusters:
	nounBucket = [wordTuple[0] for wordTuple in cluster if wordTuple[1][:2] == 'NN']
	adjBucket = [wordTuple[0] for wordTuple in cluster if wordTuple[1][:2] == 'JJ']

	# if len(nounBucket) > 10:
	# 	nounBucket = nounBucket[:10]
	# if len(adjBucket) > 10:
	# 	adjBucket = adjBucket[:10]

	bucketCluster = [adjBucket, nounBucket]
	posClusters.append(bucketCluster)

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words)
top_assoc = finder.nbest(bigram_measures.pmi, 10000)

topics = []
blacklist = {}

for assoc in top_assoc:
	word1 = assoc[0]
	word2 = assoc[1]

	for cluster in posClusters:
		adjBucket = cluster[0]
		nounBucket = cluster[1]

		if word1 in adjBucket and word2 in nounBucket:

			topicName = " ".join([word1, word2])
			queryParams = [topicName, adjBucket, nounBucket]
			if topicName not in blacklist:
				topics.append(queryParams)
			else:
				blacklist[topicName] = 1
			continue

		if word2 in adjBucket and word1 in nounBucket:
			topicName = " ".join([word2, word1])
			queryParams = [topicName, adjBucket, nounBucket]
			if topicName not in blacklist:
				topics.append(queryParams)
			else:
				blacklist[topicName] = 1
			continue

outfile = open('test.txt', 'w')

for topic in topics[:10]:
  outfile.write("%s\n" % topic[0])
  outfile.write("%s\n" % topic[1:])


sc.stop()


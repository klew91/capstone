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

review = """I went to the Nike store today because I was looking for a specific kind of basketball shoe. 
		    I wanted to get a pair of Nike Lebrons, because they are supposed to be the best basketball
		    shoe out there. I went up to the basketball shoe section and there was a worker who came and
		    offered to help me find what I was looking for. He showed me where the Lebrons were, but when
		    I told him I have large, wide feet, he recommended that I go with a different shoe because the
		    Lebrons are pretty narrow. He recommended the Kyrie Irving shoe but they didn't have it in my size.
		    Next he recommended the Air Jordans, so I went over to look at those instead. I quickly found
		    a pair that I liked that were on sale. I tried on two different sizes and the bigger one fit
		    perfectly! I decided to buy the shoes right then and there. I thought the worker did a great job
		    of finding out what I needed and helping me pick a shoe that fit my needs best."""

tokenizer = RegexpTokenizer(r'\w+')

def read_words(words_file):
    return [word for line in open(words_file, 'r') for word in tokenizer.tokenize(line)]
#words = tokenizer.tokenize(review)
# [word for sent in sent_tokenize(review) for word in word_tokenize(sent)]

words = read_words('data.txt')
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if not w in stop_words]

sc = SparkContext(appName="KeywordCluster")
filtered_words = sc.parallelize(filtered_words)

counts = filtered_words.map(lambda x: (x, 1)) \
		.reduceByKey(add) \
		.sortBy(lambda x: x[1], ascending=False)

#def print_freq(w):
#	print(word_frequency(w[0], 'en') * 1e6)
#counts.foreach(print_freq)

def calc_SIP(wordTuple):
	word = wordTuple[0]
	freq = word_frequency(word, 'en')
	if freq == 0:
		freq = float(.00001)
	SIPscore = wordTuple[1]/freq
	return (word, SIPscore)

SIPscores = counts.map(calc_SIP)\
				  .filter(lambda x: x[1] > 0.7)\
				  .sortBy(lambda x: x[1], ascending=False)

#SIPscores.saveAsTextFile(sys.argv[1])

sentences = [words]
model = Word2Vec(sentences, size=300, window=2, min_count=0, workers=4)
#print(model.wv.similarity('dog', 'wolf'))

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
				if model.wv.similarity(parent_word[0], word[0]) > 0.95:
					blacklist[word[0]] = 1
					cluster.append(word[0])
		cluster = pos_tag(cluster)
		clusters.append(cluster)

posClusters = []
for cluster in clusters:
	nounBucket = [wordTuple[0] for wordTuple in cluster if wordTuple[1][:2] == 'NN']
	adjBucket = [wordTuple[0] for wordTuple in cluster if wordTuple[1][:2] == 'JJ']
	bucketCluster = [adjBucket, nounBucket]
	posClusters.append(bucketCluster)

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words)
top_assoc = finder.nbest(bigram_measures.pmi, 10000)

topics = {}

for assoc in top_assoc:
	word1 = assoc[0]
	word2 = assoc[1]

	for cluster in posClusters:
		adjBucket = cluster[0]
		nounBucket = cluster[1]

		if word1 in adjBucket and word2 in nounBucket:

			topicName = " ".join([word1, word2])
			queryParams = [adjBucket, nounBucket]
			if topicName in topics:
				topics[topicName] += queryParams
			else:
				topics[topicName] = queryParams
			continue

		if word2 in adjBucket and word1 in nounBucket:
			topicName = " ".join([word2, word1])
			queryParams = [adjBucket, nounBucket]
			if topicName in topics:
				topics[topicName] += queryParams
			else:
				topics[topicName] = queryParams
			continue


outfile = open('test.txt', 'w')

for topic in topics:
  outfile.write("%s\n" % topic)
  outfile.write("%s\n" % topics[topic])


sc.stop()


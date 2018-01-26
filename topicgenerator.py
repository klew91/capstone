from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import sys
from operator import add
from pyspark import SparkContext
import re
from wordfreq import word_frequency
import collections
from gensim.models import Word2Vec

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
    return [word for line in open(words_file, 'r') for word in line.split()]
#words = tokenizer.tokenize(review)
# [word for sent in sent_tokenize(review) for word in word_tokenize(sent)]

words = read_words('warworlds.txt')
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

SIPscores = counts.map(lambda x: (x[0], x[1]/(word_frequency(x[0], 'en')
* 1e6 + .00000000000000001)))

#SIPscores.saveAsTextFile(sys.argv[1])
sc.stop()

sentences = [words]
model = Word2Vec(sentences, size=100, window=5, min_count=0, workers=4)
#print(model.wv.similarity('dog', 'wolf'))
#print(model.wv.similarity('forest', 'light'))
#print(model.wv.similarity('dog', 'snarl'))
print(model.wv.similarity('dog', 'man'))
#print(model.wv.similarity('dog', 'sled'))
print(model.wv.similarity('man', 'food'))
print(model.wv.similarity('man', 'woman'))



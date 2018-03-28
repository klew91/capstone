
# coding: utf-8

# ### Project Dependencies

# In[2]:


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

import unidecode
import csv


# ### Import and Clean Data

# In[3]:


reviews = []
freviews = []
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
with open('satisfaction_clean.csv') as inputfile:
    for line in inputfile:
        line_words = tokenizer.tokenize(unidecode.unidecode(line).lower())
        reviews.append(line_words)
        freviews.append([w for w in line_words if not w in stop_words])
words = [word for line in reviews for word in line]
fwords = [word for line in freviews for word in line]
num_words = len(words)
num_fwords = len(fwords)

# remove empty reviews
reviews = [x for x in reviews if x != []]
freviews = [x for x in freviews if x != []]


# In[4]:


print(len(reviews))
print(len(freviews))
print(len(words))
print(len(fwords))


# ### SIP Scores

# In[42]:


text = nltk.Text(fwords)
uniq_fwords = list(set(fwords))
word_pool = []
word_count = []
word_SIP = []
for word in uniq_fwords:
    c = text.count(word)
    if c < 6:
        continue
    word_pool.append(word)
    freq = word_frequency(word, 'en')
    if freq == 0:
        freq = float(.00001)
    word_SIP.append(c/num_fwords/freq)
    word_count.append(c)

word_pool_sorted = [x for _,x in sorted(zip(word_SIP, word_pool), reverse = True)]
word_count_sorted = [x for _,x in sorted(zip(word_SIP, word_count), reverse = True)]
word_SIP_sorted = word_SIP
word_SIP_sorted.sort(reverse = True)


# In[43]:


print(word_pool_sorted[:10])
print(word_count_sorted[:10])
print(word_SIP_sorted[:10])


# ### Word2Vec

# In[44]:


model = Word2Vec(freviews, size=300, window=5, min_count=1, workers=4)


# In[45]:


print(model.wv.most_similar(positive=['qualtrics'], topn=5))
print(model.wv.most_similar(positive=['surveys'], topn=5))


# ### Separate Nouns and Adjectives

# In[46]:


# this will keep the nouns and adjs in order of SIP score
word_pos = word_pool_sorted
word_pos = pos_tag(word_pos)


# In[47]:


word_pos[:10]


# In[48]:


nounBucket = []
adjBucket = []
# for now, just call all non-nouns "adjectives"
for word in word_pos:
    if word[1][:2] == "NN":
        nounBucket.append(word[0])
    else:
        adjBucket.append(word[0])

print(nounBucket[:10])
print(adjBucket[:10])


# 
# ### Cluster the Nouns

# In[49]:


wordBank = nounBucket

num_clust = 13
similarity_thresh = 0.60
SIP_thresh = 2.0

nounClusters = []
for i in range(num_clust):
    parent_word = wordBank[0] # the first one has highest SIP
    cluster = []
    cluster.append(parent_word)
    count = 0
    for i,word in enumerate(wordBank[1:]):
        if count < 9:
            if model.wv.similarity(parent_word, word) > similarity_thresh:
                if word_SIP_sorted[i+1] > SIP_thresh:
                    cluster.append(word)
                    count += 1
        else:
            break
    nounClusters.append(cluster)
    wordBank = [t for t in wordBank if t not in cluster] # remove current clustr from wordBank


# In[50]:


nounClusters[:15]


# ### Find Related Words from Adjective Bucket

# In[51]:


adjClusters = []
bigram_measures = nltk.collocations.BigramAssocMeasures()
adjSet = set(adjBucket)
num_adj = 10
for cluster in nounClusters:
    # make a copy of reviews, replace all cluster words with the parent_word of that cluster
    parent_word = cluster[0]
    similar_words = cluster[1:]
    words_copy = [parent_word if w in similar_words else w for w in words]
    
    # now find bigrams with parent_word and all words in adjBucket
    finder = BigramCollocationFinder.from_words(words_copy, window_size=5)
    parent_filter = lambda *w: parent_word not in w
    adj_filter = lambda w1, w2: adjSet.isdisjoint([w1, w2])
    finder.apply_freq_filter(2)
    finder.apply_ngram_filter(parent_filter)
    finder.apply_ngram_filter(adj_filter)
    adj_temp = finder.nbest(bigram_measures.pmi, num_adj)
    adj_temp = [pair[1] if pair[0] == parent_word else pair[0] for pair in adj_temp]
    adjClusters.append(adj_temp)


# In[52]:


adjClusters[:10]


# ### Save Topics

# In[53]:


topics = []
for i in range(num_clust):
    if len(adjClusters[i]) != 0:
        topicName = " ".join([nounClusters[i][0], adjClusters[i][0]])
        queryParams = [topicName, adjClusters[i][1:], nounClusters[i][1:]]
    else:
        topicName = nounClusters[i][0]
        queryParams = [topicName, [], nounClusters[i][1:]]
    topics.append(queryParams)


# In[54]:


topics


# In[56]:


outfile = open('topicsPOS.txt', 'w')

for topic in topics[:10]:
  outfile.write("%s\n" % topic[0])
  outfile.write("%s\n" % topic[1:])


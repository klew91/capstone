
# coding: utf-8

# ### Project Dependencies

# In[1]:


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
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn


# ### Load and Format Data

# In[2]:


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
stop_words.add('sanitized')
filtered_words = [w for w in words if not w in stop_words]


# In[3]:


print(len(words))
print(len(filtered_words))


# ### Combine Frequent Bigrams

# In[4]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(filtered_words, window_size=2)
finder.apply_freq_filter(3)
top_bigrams = finder.nbest(bigram_measures.likelihood_ratio, 20)
# print(top_bigrams)


# In[5]:


combined_words = [w for w in words if not w in stop_words]
combined_words = filtered_words
numB = len(filtered_words) - 1
count = 0
for i in range(numB):
    bigram = (filtered_words[i], filtered_words[i+1])
    if bigram in top_bigrams:
        j = i - count
        k = (i + 1) - count
        combined_words[j:k] = ['_'.join(bigram)]
        count = count + 1


# In[6]:


combined_words[:10]


# ### SIP Scores
# An alternative method that is a bit faster for smaller datasets than using Spark. Note that dividing by "num_words" normalizes the SIP score, so that any word with SIP = 1 is used in the corpus just as often as in normal english language. A word with SIP = 5 is used 5 times as often in the corpus as it is in normal english.

# In[7]:


num_words = len(combined_words)
text = nltk.Text(combined_words)
uniq_words = list(set(combined_words))
word_pool = []
word_count = []
word_SIP = []
for word in uniq_words:
    c = text.count(word)
    # only interested in words that occur more than 5 times
    if c < 6:
        continue
    word_pool.append(word)
    freq = word_frequency(word, 'en')
    if freq == 0:
        freq = float(.00001)
    # normalize the SIP scores
    word_SIP.append(c/num_words/freq)
    word_count.append(c)

word_pool_sorted = [x for _,x in sorted(zip(word_SIP, word_pool), reverse = True)]
word_count_sorted = [x for _,x in sorted(zip(word_SIP, word_count), reverse = True)]
word_SIP_sorted = word_SIP
word_SIP_sorted.sort(reverse = True)
SIPscoreslist = list(zip(word_pool_sorted, word_SIP_sorted))


# In[8]:


SIPscoreslist[:15]


# ### Word2Vec
# Word2Vec feels very unstable to me, at least with this few responses. 

# In[9]:


sentences = [combined_words]
model = Word2Vec(sentences, size=300, window=2, min_count=1, workers=4)


# In[10]:


# most similar words according to Word2Vec
print(model.wv.most_similar(positive=['qualtrics'], topn=5))
print(model.wv.most_similar(positive=['easy_use'], topn=5))
print(model.wv.most_similar(positive=['tool'], topn=5))


# In[11]:


print(model.wv.similarity('customer_support', 'customer_service'))
print(model.wv.similarity('customer', 'support'))
print(model.wv.similarity('affordable', 'cost'))
print(model.wv.similarity('cost','price'))


# ### Stem Words

# In[12]:


stemmer = SnowballStemmer('english')
stem_words = [stemmer.stem(w) for w in word_pool_sorted]


# ### Synonymns

# In[13]:


# this takes longer...so let's limit to top 30
NUM_TOPICS = 30
top_words = word_pool_sorted[:NUM_TOPICS]
synonyms = []
for word in top_words:
    word_synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            word_synonyms.append(l.name())
    synonyms.append(list(set(word_synonyms[:4])))


# In[14]:


synonyms


# ### Base "Noun" Clusters

# In[15]:


clusters = []
blacklist = {}
for i, parent_word in enumerate(SIPscoreslist):
    if parent_word[0] not in blacklist:
        # this is trying to account for "survey" and "surveys"
        if stemmer.stem(parent_word[0]) not in blacklist:
            
            # begin the cluster with parent_word
            cluster = []
            blacklist[parent_word[0]] = 1
            cluster.append(parent_word[0])
            
            # add 5 similar words from Word2Vec
            top_sim = [p[0] for p in model.wv.most_similar(positive=parent_word[0], topn=20)] # top 20
            top_sim = [w for w in top_sim if w not in blacklist] # remove any in blacklist
            top_sim = [w for w in top_sim if stemmer.stem(w) not in blacklist]
            for word in top_sim[:5]: # take the top 5 not in blacklist
                cluster.append(word)
                blacklist[word] = 1
            
            # add 4 similar words from WordNet synonyms
            top_sim = synonyms[i]
            for word in top_sim:
                if word not in blacklist:
                    if stemmer.stem(word) not in blacklist:
                        cluster.append(word)
                        blacklist[word] = 1
            
            # finish the cluster
            clusters.append(cluster)
    # force it to stop if we run past NUM_TOPICS
    if i == NUM_TOPICS - 1:
        break


# In[36]:


# print(model.wv.most_similar(positive=['qualtrics'], topn=10))
# print(model.wv.most_similar(positive=['user_friendly'], topn=10))
# print(model.wv.most_similar(positive=['great_tool'], topn=10))


# In[16]:


print(len(clusters))
print(clusters[0])
print(clusters[1])
print(clusters[2])
print(clusters[3])
print(clusters[10])


# ### Descriptive "Adjective" Clusters

# In[17]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
descriptions = []

# this is a bigram filter to remove bigrams with words in the blacklist
def create_myfilter(parent):
    def bigram_filter(w1, w2):
        if w1 == parent:
            return w2 in blacklist
        if w2 == parent:
            return w1 in blacklist
    return bigram_filter

# for every main/noun cluster....
for i, cluster in enumerate(clusters):
    # make a copy of the text, and replace 
    # all cluster words with the parent_word of that cluster
    parent_word = cluster[0]
    similar_words = cluster[1:]
    words_copy = [parent_word if w in similar_words else w for w in combined_words]
    
    # find top 10 bigrams containing the parent_word
    # and NOT contains blacklist words
    finder = BigramCollocationFinder.from_words(words_copy, window_size=5)
    parent_filter = lambda *w: parent_word not in w   
    blacklist_filter = create_myfilter(parent_word)
    finder.apply_ngram_filter(parent_filter)      # bigram must contain parent_word
    finder.apply_ngram_filter(blacklist_filter)   # bigram does not contain blacklist words
    finder.apply_freq_filter(3)                   # bigram occurs at least 3 times
    best_bigrams = finder.nbest(bigram_measures.likelihood_ratio, 10)
    adj = []
    list(adj.extend(row) for row in best_bigrams)
    l = [w for w in adj if w != parent_word]
    descriptions.append(adj)

for i,cluster in enumerate(clusters):
    descriptions[i] = [w for w in descriptions[i] if w != cluster[0]]


# In[18]:


print(descriptions[0])
print(descriptions[1])
print(descriptions[2])


# ### Cluster Titles

# In[19]:


def create_clustfilter(clust):
    def bigram_filter(w1, w2):
        return w2 not in clust and w1 not in clust
    return bigram_filter

topics = []
for i in range(len(clusters)):
    pool = clusters[i] + descriptions[i]
    finder = BigramCollocationFinder.from_words(combined_words, window_size=2)
    clust_filter = create_clustfilter(pool)  
    topic_filter = lambda w1, w2: (w1, w2) in set(topics)
    finder.apply_ngram_filter(clust_filter)
    finder.apply_ngram_filter(topic_filter)
    topics.append(finder.nbest(bigram_measures.likelihood_ratio, 1)[0])


# In[20]:


topics[:10]


# ### Save Topics

# In[32]:


final_topics = []
for i,topic in enumerate(topics):
    topicName = " ".join(topic)
    clust = clusters[i]
    adj = descriptions[i]
    queryParams = [topicName, clust, adj[1:]]
    final_topics.append(queryParams)


# In[33]:


final_topics[:2]


# In[35]:


outfile = open('topicsKaite.txt', 'w')

for topic in final_topics[:10]:
    outfile.write("%s\n" % topic[0])
    outfile.write("%s\n" % topic[1:])


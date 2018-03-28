
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
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer


# ### Import and Clean Data

# In[2]:


reviews = []
freviews = []
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.add('sanitized')
snowball_stemmer = SnowballStemmer('english')

with open('satisfaction_clean.csv') as inputfile:
    for line in inputfile:
        line_words = tokenizer.tokenize(unidecode.unidecode(line).lower())
        reviews.append(line_words)
        freviews.append([w for w in line_words if not w in stop_words])
words = [word for line in reviews for word in line]
fwords = [word for line in freviews for word in line]
swords = [snowball_stemmer.stem(w) for w in fwords]
num_words = len(words)
num_fwords = len(fwords)
num_swords = len(swords)

# remove empty reviews
reviews = [x for x in reviews if x != []]
freviews = [x for x in freviews if x != []]

# fix some custom bugs
fwords = [x if x != 'tool' else 'instrument' for x in fwords]
fwords = [x if x != 'tools' else 'instruments' for x in fwords]


# In[17]:


print(len(reviews))
print(len(freviews))
print(len(words))
print(len(fwords))
print(len(swords))


# ### SIP Scores
# This is an alternative way to calculate the SIP scores. It is a bit faster for smaller datasets than using Spark.
# 
# I divide the count "c" by the total number of filtered words in the corpus "num_fwords" and THEN I divide by the frequency. This normalizes the SIP scores, so that words with SIP = 1 occur just as often in the corpus as they do in normal english language. Any word with SIP > 1 occurs more frequently. I chose a cutoff value of 5, so we are only looking at words that occur 5 times as often as they do in normal english.

# In[3]:


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


# In[19]:


print(word_pool_sorted[:40])
print(word_count_sorted[:30])
print(word_SIP_sorted[:10])


# ### Find Synonyms for Parent Words
# The words with the highest SIP scores become the parent words for our topics. For each word, we are pulling the synonyms from a WordNet dictionary to form a cluster.

# In[4]:


# top 20 words
top_words = word_pool_sorted[:20]
synonyms = []
for word in top_words:
    word_synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            word_synonyms.append(l.name())
    synonyms.append(list(set(word_synonyms)))


# In[21]:


print(top_words[:5])
print(synonyms[:5])


# ### Find Bigrams Containing Parent Words
# Now, we take each of our parent words and find the five top (as measured by PMI) bigrams containing that word. These bigrams will be used as the topic titles. 

# In[5]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
all_bigrams = []
for i in range(len(top_words)):
    # make a copy of the text, and replace 
    # all cluster words with the parent_word of that cluster
    parent_word = top_words[i]
    similar_words = synonyms[i]
    fwords_copy = [parent_word if w in similar_words else w for w in fwords]
    
    # find top five bigrams containing the parent_word
    # note: using a window size = 3 means the bigrams do not 
    #       have to be directly next to each other
    finder = BigramCollocationFinder.from_words(fwords_copy, window_size=5)
    parent_filter = lambda *w: parent_word not in w   
    finder.apply_ngram_filter(parent_filter)          # bigram must contain parent_word
    finder.apply_freq_filter(3)                       # bigram must occur at least three times
    temp = finder.nbest(bigram_measures.likelihood_ratio, 1)       # keep the top 5 bigrams
    all_bigrams.append(temp)


# In[27]:


# some good examples of topic bigrams: "qualtrics affordable", "surveys monkey", 
#                                      "intuitive platform", "responsive customer", "ease learning"
print(all_bigrams[:10])
print(len(top_words))
print(len(all_bigrams))
print(len(all_bigrams[0]))


# ### Find Synonyms for Topics
# The parent words already have synonym clusters, but now we need to find the synonyms for the other part of the bigrams.

# In[6]:


topics = []
topic_clusters = []
# loop through top_words (aka: words with high SIP scores)
for i in range(len(top_words)):
    word = top_words[i]
    # loop through all the bigrams created for that specific word
    for pair in all_bigrams[i]:
        if pair[0] == word:
            # find the other syn cluster
            s2 = []
            for syn in wn.synsets(pair[1]):
                for l in syn.lemmas():
                    s2.append(l.name())
            # make the topic and topic clusters
            topics.append(pair)
            topic_clusters.append([synonyms[i], list(set(s2))])
        elif pair[1] == word:
            # find the other syn cluster
            s2 = []
            for syn in wn.synsets(pair[0]):
                for l in syn.lemmas():
                    s2.append(l.name())
            # make the topic and topic clusters
            topics.append(pair)
            topic_clusters.append([list(set(s2)), synonyms[i]])
        else:
            print('ERROR: Bigram does not contain any top words.')
            break


# In[7]:


topics[0:25]


# In[8]:


topic_clusters[:5]


# In[9]:


topics[10]


# In[34]:


topic_clusters[10]


# 
# ### Search Reviews

# In[35]:


THE_TOPIC = 5
printr = []
for i in range(len(reviews)):
    if not set(topic_clusters[THE_TOPIC][0]).isdisjoint(reviews[i]):
        if not set(topic_clusters[THE_TOPIC][1]).isdisjoint(reviews[i]):
            printr.append(i)
print(len(printr))


# In[36]:


for i in printr:
    print(reviews[i])


# ### Save Topics

# In[10]:


final_topics = []
for i in range(len(topics)):
    topicName = " ".join([topics[i][0], topics[i][1]])
    queryParams = [topicName, topic_clusters[i][0], topic_clusters[i][1]]
    final_topics.append(queryParams)


# In[11]:


final_topics


# In[14]:


outfile = open('topicsBigrams.txt', 'w')

for topic in final_topics[:10]:
  outfile.write("%s\n" % topic[0])
  outfile.write("%s\n" % topic[1:])


from nltk.stem import SnowballStemmer

### F1 Score
# takes two lists of words (noun and adj clusteres combined)
def f1_score (topicHuman, topicComputer):
    # stemming
    # note: lemmentizing might work better?? but it is slower, 
    #       and may not work for weird words like "qualtrics"?
    snowball_stemmer = SnowballStemmer('english')
    topicHuman = [snowball_stemmer.stem(w) for w in topicHuman]
    topicComputer = [snowball_stemmer.stem(w) for w in topicComputer]
    
    # convert to sets - gets rid of duplicates and makes it easier to find common words
    topicHuman = set(topicHuman)
    topicComputer = set(topicComputer)
    
    # calculate F1 score
    total_truth = len(topicHuman)
    total_terms = len(topicComputer)
    total_right = len(topicHuman & topicComputer)
    precision = total_right / total_terms
    recall = total_right / total_truth
    F1 = 2 / ((1/precision) + (1/recall))
    return F1

### Example 
# noun and adj clusters combined
topic1 = ['shoe','boots','sneakers','slipper','fit','tight','loose','comfortable','size','hurt']
topic2 = ['shoe','shoes','boot','slipper','sneaker','sandal','fit','tight','loose']

# note: it doesn't matter what order you do
print(f1_score(topic1, topic2))
print(f1_score(topic2, topic1))

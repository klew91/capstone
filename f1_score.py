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

    if precision == 0:
        precision = float(.00001)
    if recall == 0:
        recall = float(.00001)

    F1 = 2 / ((1/precision) + (1/recall))
    return F1

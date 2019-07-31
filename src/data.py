from csv import reader
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np

def append_sentences(path):
    target = []
    with open(path, newline = '') as sentences:                                                                                          
        sentence_reader = reader(sentences, delimiter='\n')    
        for sentence in sentence_reader:
            target.append(sentence[0])
    return target

definition_examples = append_sentences('../data/definition_examples.txt')
definition_negative_examples= append_sentences('../data/definition_negative_examples.txt')
defintion_other = append_sentences('../data/definition_other.txt')

train_sentences = np.append(definition_examples, definition_negative_examples)    

vectorizer = CountVectorizer()  
train_arr = vectorizer.fit_transform(train_sentences).toarray()
test_arr = vectorizer.transform(defintion_other).toarray()
features = vectorizer.get_feature_names()

def prepare_data(feature_lists, example_count):
    target = []
    for x in range(len(feature_lists)):
        word_dict = dict(zip(features, feature_lists[x]))
        word_dict["is_example"] = True if x < example_count else False
        target.append(word_dict)
    return target

train_data = prepare_data(train_arr,len(definition_examples))
test_data = prepare_data(test_arr,len(test_arr)/2)
#TODO: make length of data variable

found_max = 0
for values in train_arr:
    cur_max = max(values)
    if cur_max > found_max: found_max = cur_max

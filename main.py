import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import uuid

import pyro
import pyro.distributions as dist

from sklearn.feature_extraction.text import CountVectorizer  

pyro.set_rng_seed(42)

definition_examples = []
definition_negative_examples = []
defintion_other = []

with open('definition_examples.txt', newline = '') as definitions:                                                                                          
    definition_reader = csv.reader(definitions, delimiter='\n')    
    for definition in definition_reader:
        definition_examples.append(definition[0])

with open('definition_negative_examples.txt', newline = '') as definitions:                                                                                          
    definition_reader = csv.reader(definitions, delimiter='\n')    
    for definition in definition_reader:
        definition_negative_examples.append(definition[0])
  
with open('definition_other.txt', newline = '') as definitions:                                                                                          
    definition_reader = csv.reader(definitions, delimiter='\n')    
    for definition in definition_reader:
        defintion_other.append(definition[0])    
    
vectorizer = CountVectorizer()  
sentences = np.append(definition_examples, definition_negative_examples)
X = vectorizer.fit_transform(sentences) 

features = vectorizer.get_feature_names()

data = []
all_objs = []
X_arr = X.toarray()
for x in range(len(X_arr)):
    word_dict = dict(zip(features, X_arr[x]))
    word_dict["is_example"] = True if (x < (len(defintion_other)/2) or (x >= len(defintion_other) and x < (len(X_arr)-len(definition_negative_examples)))) else False
    all_objs.append(word_dict)
    data.append(word_dict)

other_obs = vectorizer.transform(defintion_other).toarray()
other_obs = list(map(lambda obs : dict(zip(features, obs)), other_obs))

for x in range(len(defintion_other)):
    other_obs[x]["is_example"] = True if x < 4 else False
all_objs = np.append(all_objs, other_obs)

max_feature_value = 0
for values in X.toarray():
    found_max = max(values)
    max_feature_value = found_max if found_max > max_feature_value else max_feature_value

def uniform_draw(object_list):
    probs = []
    prob = 1/len(object_list)
    for obj in object_list:
            probs.append(prob)
    sample_name = str(uuid.uuid1())        
    sample = pyro.sample(sample_name, dist.Categorical(probs=torch.Tensor(probs)))
    return object_list[sample]

value_list = [x for x in range(found_max+1)]
    

def sample_pred():
    feature = uniform_draw(features)
    value = uniform_draw(value_list)
    return lambda x : x[feature] == value

tau = 0.3;
noiseParam = np.exp(-1.5)

def sample_conj():
    sample_name = str(uuid.uuid1())        
    if pyro.sample(sample_name, dist.Bernoulli(tau)):
        c = sample_conj()
        p = sample_pred()
        return lambda x : c(x) and p(x)
    else: 
        return sample_pred()

def get_formula():
    sample_name = str(uuid.uuid1())        
    if pyro.sample(sample_name, dist.Bernoulli(tau)):
        c = sample_conj()
        f = get_formula()
        return lambda x : c(x) or f(x)
    else: 
        return sample_conj()
get_formula()

def model():
    rule = get_formula()
    
    obs_fn = lambda datum : pyro.sample("obs_fn", 
                                       dist.Bernoulli(rule(datum) if 1-noiseParam else noiseParam, obs=datum["is_example"]))

    map(obs_fn,data)
    return rule

accuracy = 0
rule = model()
choices = []
for sentence in all_objs:
    if rule(sentence) == sentence["is_example"]: accuracy += 1

accuracy /= len(all_objs)
accuracies = [accuracy]
print("Initial accuracy " + str(accuracy))

tries = 5000

for n in range(tries):
    cur_accuracy = 0
    cur_rule = model()
    cur_choices = []
    for sentence in other_obs:
        choice = cur_rule(sentence)
        if choice == sentence["is_example"]: cur_accuracy += 1
        cur_choices.append(choice)
    cur_accuracy /= len(other_obs)
    accuracies.append(cur_accuracy)
    #print(cur_accuracy)
    if cur_accuracy>accuracy:
        rule = cur_rule
        accuracy = cur_accuracy
        choices = cur_choices

#print(accuracies)
print("Final accuracy " + str(accuracy))
print("Choices are " + str(choices))

test_str = "Machine Translation is used for NLP"
word_dict = vectorizer.transform([test_str])
word_dict = dict(zip(features, word_dict.toarray()[0]))
word_dict["is_example"] = True
#print(rule(word_dict))
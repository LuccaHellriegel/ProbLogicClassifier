import uuid
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from data import SentenceData

tau = 0.3
noiseParam = np.exp(-1.5)
complexity_lower_bound = 2

sentenceData = None
value_list = None

global rule_string
rule_string = ""
global rule_complexity
rule_complexity = 0

pyro.set_rng_seed(42)

def set_seed(n):
    pyro.set_rng_seed(n)

def refresh_data():
    global sentenceData
    sentenceData = SentenceData()
    global value_list
    value_list = [x for x in range(sentenceData.found_max+1)]

def uniform_draw(object_list):
    probs = []
    prob = 1/len(object_list)
    for obj in object_list:
            probs.append(prob)
    sample_name = str(uuid.uuid1())        
    sample = pyro.sample(sample_name, dist.Categorical(probs=torch.Tensor(probs)))
    return object_list[sample]

def sample_pred():
    feature = uniform_draw(sentenceData.features)
    value = uniform_draw(value_list)
    global rule_string
    rule_string += "(x[" + feature + "]==" + str(value) + ")"
    global rule_complexity
    rule_complexity += 1 
    return lambda x : x[feature] == value

def sample_conj():
    sample_name = str(uuid.uuid1())        
    if pyro.sample(sample_name, dist.Bernoulli(tau)):
        global rule_string
        rule_string += "{and"
        global rule_complexity
        rule_complexity += 2
        c = sample_conj()
        p = sample_pred()
        rule_string += "}"
        return lambda x : c(x) and p(x)
    else: 
        return sample_pred()

def get_formula():
    sample_name = str(uuid.uuid1())        
    if pyro.sample(sample_name, dist.Bernoulli(tau)):
        global rule_string
        rule_string += "{or"
        global rule_complexity
        rule_complexity += 2
        c = sample_conj()
        f = get_formula()
        rule_string += "}"
        return lambda x : c(x) or f(x)
    else: 
        return sample_conj()

def rule_generator():
    rule = get_formula()
    obs_fn = lambda datum : pyro.sample("obs_fn", 
                                       dist.Bernoulli(1-noiseParam 
                                       if rule(datum) else noiseParam, obs=datum["is_example"]))
    map(obs_fn,sentenceData.train_data)
    return rule

def search_rule(n):
    accuracy = 0
    rule = rule_generator()
    global rule_string
    final_rule_string = rule_string  
    global rule_complexity
    final_rule_complexity = rule_complexity
    for i in range(n):
        rule_complexity = 0
        rule_string = ""
        cur_accuracy = 0
        cur_rule = rule_generator()
        cur_rule_string = rule_string
        for sentence in sentenceData.train_data:
            if cur_rule(sentence) == sentence["is_example"]: cur_accuracy += 1
        cur_accuracy /= len(sentenceData.train_data)
        print("Current rule has the training accuracy: "+str(cur_accuracy))
        if cur_accuracy>accuracy and rule_complexity >= complexity_lower_bound:
            rule = cur_rule
            accuracy = cur_accuracy
            final_rule_string = cur_rule_string
            final_rule_complexity = rule_complexity
    print("Training accuracy of the final rule is: "+str(accuracy))
    rule_string = final_rule_string
    format_rule_string()
    print("Final rule is: " + rule_string)
    rule_complexity = final_rule_complexity
    print("Final rule complexity is: " + str(rule_complexity))
    return accuracy, rule

def insert_newline(index):
    global rule_string
    rule_string = rule_string[:index] + '\n' + rule_string[index:]

def insert_empty(index, number):
    empty_str = ""
    for i in range(number):
        empty_str += " "
    global rule_string
    rule_string = rule_string[:index] + empty_str + rule_string[index:]

def format_rule_string():
    if rule_string[0] == "(": return
    positions = []
    lengths = []
    if rule_string[1] == "a":
        positions.append(4)
        lengths.append(4+14)
    else:
        positions.append(3)
        lengths.append(3+14)
    for index in range(positions[0],len(rule_string)):
        if rule_string[index] == "{":
             if rule_string[index-1] != "d" and rule_string[index-1] != "r":
                positions.append(index+sum(lengths)+len(lengths))
                lengths.append(index+1+14)
             if rule_string[index+1] == "a":
                positions.append(index+4+sum(lengths)+len(lengths))
                lengths.append(index+1+14+3)
             else:
                positions.append(index+3+sum(lengths)+len(lengths))
                lengths.append(index+1+14+2)
    for index in range(len(lengths)):
        insert_newline(positions[index])       
        insert_empty(positions[index]+1,lengths[index])
        
                                   

def test_rule(rule):
    accuracy = 0
    choices = []
    for sentence in sentenceData.test_data:
        choice = rule(sentence)
        if choice == sentence["is_example"]: accuracy += 1
        choices.append(choice)
    accuracy /= len(sentenceData.test_data)    
    print("Testing accuracy of the final rule is: " + str(accuracy))
    print("Rule-decision: Is definition? " + str(choices))    
    return accuracy, choices

import uuid
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from data import SentenceData

tau = 0.3
noiseParam = np.exp(-1.5)
sentenceData = SentenceData()
value_list = [x for x in range(sentenceData.found_max+1)]

#TODO: make seed variable?
pyro.set_rng_seed(42)

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
    return lambda x : x[feature] == value

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

def model():
    rule = get_formula()
    obs_fn = lambda datum : pyro.sample("obs_fn", 
                                       dist.Bernoulli(1-noiseParam 
                                       if rule(datum) else noiseParam, obs=datum["is_example"]))
    map(obs_fn,sentenceData.train_data)
    return rule

def search_rule(n):
    accuracy = 0
    choices = []
    rule = model()
    for i in range(n):
        cur_accuracy = 0
        cur_rule = model()
        cur_choices = []
        for sentence in sentenceData.test_data:
            choice = cur_rule(sentence)
            if choice == sentence["is_example"]: cur_accuracy += 1
            cur_choices.append(choice)
        cur_accuracy /= len(sentenceData.test_data)
        if cur_accuracy>accuracy:
            rule = cur_rule
            accuracy = cur_accuracy
            choices = cur_choices
    print("Final accuracy " + str(accuracy))
    print("Choices are " + str(choices))
    return accuracy, choices

#TODO use not 1/0 logic but count or separated clauses (?) for percentage ranking

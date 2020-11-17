import math
from pathlib import Path

import torch
from torch import nn
import numpy as np
import checklist
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.pred_wrapper import PredictorWrapper
from checklist.expect import Expect
import numpy as np
import spacy
import re
from pattern.en import sentiment
# from transformers import pipeline
# classifier = pipeline('sentiment-analysis')
# results = classifier('We are very happy to include pipeline into the transformers repository.')
# print(results)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'

class NeuralModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    def run_example(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            # ((negative_logit, positive_logit), ([bs, sent_len, dim]x7))
            # Outputs is a tuple has two items, sentiment predictions and neural network intermediate values
            # Those values consist of seven tensors (matrix), from layer 0 (can be ignored) to layer 6 (last layer, important!)
            # The size of each matrix is num_of_sentences (1) by maximum_sentence_length by number_of_neuron (768)
            outputs = self.model(**inputs, output_hidden_states=True)
            predictions = outputs[0].cpu().numpy()
            # Return only last layer states of the [CLS] token for now.
            output_state = outputs[1][-1][0, 0].cpu().numpy()
        # This converts the prediction (logits) to probability (sum to 1)
        # prob_negative,prob_positive
        scores = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        return scores, output_state

TEST_SPACE = (768,)

class DeepxploreCoverage:
    def __init__(self, threshold = 1.0):
        self.cov_map = np.zeros(TEST_SPACE, dtype=np.bool)
        self.threshold = threshold
    
    def get_cov_inc(self, new_states: np.ndarray):
        example_cov = np.absolute(new_states) > self.threshold
        new_cov_map = np.logical_or(self.cov_map, example_cov)
        cov_inc = new_cov_map.sum() - self.cov_map.sum()
        self.cov_map = new_cov_map
        return cov_inc


def Perturb_add_typo(dataset):
    return Perturb.perturb(dataset, Perturb.add_typos)

def Perturb_change_names(pdataset):
    return Perturb.perturb(pdataset, Perturb.change_names)

def Perturb_change_location(pdataset):
    return Perturb.perturb(pdataset, Perturb.change_location)

def Perturb_change_number(pdataset):
    return Perturb.perturb(pdataset, Perturb.change_number)

def Perturb_punctuation(pdataset):
    return Perturb.perturb(pdataset, Perturb.punctuation)

def Perturb_add_negation(pdataset):
    return Perturb.perturb(pdataset, Perturb.add_negation)

def Perturb_add_negation_phrase(dataset):
    def add_negative_phrase(x):
        phrases = ['Anyway, I thought it was bad.', 'Having said this, I hated it', 'The director should be fired.']
        return ['%s %s' % (x, p) for p in phrases]
    return Perturb.perturb(dataset, add_negative_phrase)

def run_perburtation(perturb, expect_bahavior):
    for test_pair in perturb.data:
        print ("Pass(1) or fail(0): ", oracle_test(test_pair,expect_bahavior)) #0 if metamorphic test fail, 1 if success


def oracle_test(test_pair, expect_bahavior):#return 0 if metamorphic test fail, 1 if success
    sentiment = []
    print (test_pair)
    for sentence in test_pair:
        pred, output_state = model.run_example(sentence)
        '''
        Need to add threshold?
        '''
        if pred[0][0] > pred[0][1]:
            sentiment.append(-1) #negative 
        elif pred[0][0] < pred[0][1]:
            sentiment.append(1) #positive
        else:   
            sentiment.append(0) #neutral 

    if expect_bahavior == "INV": #prediction should not change
        print ("sentiment prediction",sentiment)    
        success = all(ele == sentiment[0] for ele in sentiment) 
        return 1 if success else 0

    elif expect_bahavior == "CHANGE": #prediction should change
        print ("sentiment prediction",sentiment)   
        if sentiment[0] != sentiment[1]:
            return 1 #pass if original and negation sentence prediction change
        return 0 #fail

    elif expect_bahavior == "MONO_DEC": #prediction should change
        #p7 sentiment = sentiment of new,old pair: [new, old, new, old]
        #sentiment at even index i should not be less than i+1 
        print ("sentiment prediction",sentiment)   
        for i in range(0, len(sentiment), 2): 
            if sentiment[i] < sentiment[i+1]:
                return 0 #fail
        return 1 #pass if all pairs' is mono decreasing

if __name__ == "__main__":
    test_sent = "We are very happy to include pipeline into the transformers repository."
    test_sent_pert = "We are very sad to include pipeline into the transformers repository."

    model = NeuralModel()
    cov_metrics = DeepxploreCoverage()

    # Run first sentence
    pred, output_state = model.run_example(test_sent)
    cov_inc = cov_metrics.get_cov_inc(output_state)
    print(pred, cov_inc)

    # Run perturbed sentence
    pred, output_state = model.run_example(test_sent_pert)
    cov_inc = cov_metrics.get_cov_inc(output_state)
    print(pred, cov_inc)

    # Run first sentence again
    pred, output_state = model.run_example(test_sent)
    cov_inc = cov_metrics.get_cov_inc(output_state)
    print(pred, cov_inc)

    # Expected output
    # [[0.00218062 0.99781936]] 71
    # [[9.9954027e-01 4.5971028e-04]] 51
    # [[0.00218062 0.99781936]] 0

    #example dataset
    dataset = ['This was a very nice movie directed by John Smith.',
            'Mary Keen was brilliant!', 
            'I hated everything about New York.',
            'This movie was very bad!',
            'Jerry gave me 8 delicious apples.',
            'I really liked this movie.',
            'just bad.',
            'amazing.',
            ]

    nlp = spacy.load('en_core_web_sm')
    pdataset = list(nlp.pipe(dataset))

    editor = Editor()
    # Perturbation1-invariant: add typo eg. amazing --> mmazing
    p1 = Perturb_add_typo(dataset)
    
    #Perturbation2-invariant: change name 
    p2 = Perturb_change_names(pdataset)

    #Perturbation3-invariant: change location eg. New York --> CVille
    p3 = Perturb_change_location(pdataset)

    #Perturbation4-invariant: change numbers eg. 1 --> 3
    p4 = Perturb_change_number(pdataset)

    #Perturbation5-invariant: add or remove punctuation. (might not be very useful though)
    p5 = Perturb_punctuation(pdataset)

    #Perturbation6: add negation eg. like --> don't like 
    #negation function doesn't work on many sentences, create a seperate dataset that is application. 
    # Might drop this perturbation and use Perturbation7 instead. 
    dataset_negation = ['Mary Keen was brilliant.', 'This movie was bad.', "Jerry is amazing.", "You are my friend."] 
    pdataset_negation = list(nlp.pipe(dataset_negation))
    p6 = Perturb_add_negation(pdataset_negation)

    #Perturbation7: add negation phrase at the end of the sentence
    p7 = Perturb_add_negation_phrase(dataset)

    print ("---------------metamorphic relation 1: add typo ------------------")
    run_perburtation(p1,"INV")
    print ("---------------metamorphic relation 2: change name ------------------")
    run_perburtation(p2,"INV")
    print ("---------------metamorphic relation 3: change location ------------------")
    run_perburtation(p3,"INV")
    print ("---------------metamorphic relation 4: change numbers ------------------")
    run_perburtation(p4,"INV")
    print ("---------------metamorphic relation 5: add/remove punc ------------------")
    run_perburtation(p5,"INV")
    print ("---------------metamorphic relation 6: add negation ------------------")
    run_perburtation(p6,"CHANGE")
    print ("---------------metamorphic relation 7: add negation phrase------------------")
    run_perburtation(p7,"MONO_DEC")
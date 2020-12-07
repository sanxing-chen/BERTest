import math
from pathlib import Path
import json

import torch
from torch import nn
import numpy as np
from metamorphicTesting.generator import MetamorphicGenerator
from metamorphicTesting.tester import MetamorphicTester
import spacy
from pattern.en import sentiment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from transformers import pipeline
# classifier = pipeline('sentiment-analysis')
# results = classifier('We are very happy to include pipeline into the transformers repository.')
# print(results)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'


class NeuralModel:
    def __init__(self, cuda=True):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.to('cuda')

    def run_example(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        if self.cuda:
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to('cuda')
        with torch.no_grad():
            # ((negative_logit, positive_logit), ([bs, sent_len, dim]x7))
            # Outputs is a tuple has two items, sentiment predictions and neural network intermediate values
            # Those values consist of seven tensors (matrix), from layer 0 (can be ignored) to layer 6 (last layer, important!)
            # The size of each matrix is num_of_sentences (1) by maximum_sentence_length by number_of_neuron (768)
            outputs = self.model(**inputs, output_hidden_states=True)
            predictions = outputs[0].cpu().numpy()
            # Return only last layer states of the [CLS] token for now.
            output_state = outputs[1][-2][0, 0].cpu().numpy()
        # This converts the prediction (logits) to probability (sum to 1)
        # prob_negative,prob_positive
        scores = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        return scores, output_state


DIM = 768


class DeepxploreCoverage:
    def __init__(self, threshold=1.0):
        self.cov_map = np.zeros((DIM,), dtype=np.bool)
        self.threshold = threshold

    def get_cov_inc(self, new_states: np.ndarray, add_inc_to_cov=False):
        example_cov = np.absolute(new_states) > self.threshold
        new_cov_map = np.logical_or(self.cov_map, example_cov)
        cov_inc = new_cov_map.sum() - self.cov_map.sum()
        if add_inc_to_cov:
            self.cov_map = new_cov_map
        return cov_inc
    
    def get_current(self):
        return self.cov_map.sum()

class NearestNeighborCoverage:
    def __init__(self, threshold=8):
        self.cov_map = np.zeros((0, DIM))
        self.threshold = threshold

    def get_cov_inc(self, new_states: np.ndarray, add_inc_to_cov=False):
        new_states = new_states[np.newaxis, :]
        if self.cov_map.shape[0] == 0:
            cov_inc = 1
        else:
            # Euclidean distance
            diff = np.sqrt(((new_states - self.cov_map) ** 2).sum(axis=1))
            # print(diff)
            # if self.get_current() == 10:
            #     exit()
            cov_inc = 1 if np.all(diff > self.threshold) else 0
        if add_inc_to_cov and cov_inc:
            self.cov_map = np.concatenate([new_states, self.cov_map], axis=0)
        return cov_inc
    
    def get_current(self):
        return self.cov_map.shape[0]

class DeepgaugeCoverage:
    def __init__(self, threshold=1.0):
        self.cov_map = np.zeros((DIM,), dtype=np.bool)
        self.threshold = threshold

    def get_cov_inc(self, new_states: np.ndarray, add_inc_to_cov=False):
        example_cov = np.absolute(new_states) > self.threshold
        new_cov_map = np.logical_or(self.cov_map, example_cov)
        cov_inc = new_cov_map.sum() - self.cov_map.sum()
        if add_inc_to_cov:
            self.cov_map = new_cov_map
        return cov_inc
    
    def get_current(self):
        return self.cov_map.sum()

def simple_cov_test(model, cov_metrics):
    test_sent = "We are very happy to include pipeline into the transformers repository."
    test_sent_pert = "We are very sad to include pipeline into the transformers repository."

    # Run first sentence
    pred, output_state = model.run_example(test_sent)
    cov_inc = cov_metrics.get_cov_inc(output_state, True)
    print(pred, cov_inc)

    # Run perturbed sentence
    pred, output_state = model.run_example(test_sent_pert)
    cov_inc = cov_metrics.get_cov_inc(output_state, True)
    print(pred, cov_inc)

    # Run first sentence again
    pred, output_state = model.run_example(test_sent)
    cov_inc = cov_metrics.get_cov_inc(output_state, True)
    print(pred, cov_inc)

    # Expected output
    # [[0.00218062 0.99781936]] 71
    # [[9.9954027e-01 4.5971028e-04]] 51
    # [[0.00218062 0.99781936]] 0


def simple_perturb_test(metamorphic_tester):
    # example dataset
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

    # add typo
    p1 = MetamorphicGenerator.Perturb_add_typo(dataset)

    # change name
    p2 = MetamorphicGenerator.Perturb_change_names(pdataset)

    # change location
    p3 = MetamorphicGenerator.Perturb_change_location(pdataset)

    # change numbers
    p4 = MetamorphicGenerator.Perturb_change_number(pdataset)

    # add or remove punctuation
    p5 = MetamorphicGenerator.Perturb_punctuation(pdataset)

    # add negation
    # negation function doesn't work on many sentences, create a seperate dataset that is application.
    dataset_negation = ['Mary Keen was brilliant.', 'This movie was bad.', "Jerry is amazing.", "You are my friend."]
    pdataset_negation = list(nlp.pipe(dataset_negation))
    p6 = MetamorphicGenerator.Perturb_add_negation(pdataset_negation)

    # add negation phrase
    p7 = MetamorphicGenerator.Perturb_add_negation_phrase(dataset)

    print("---------------metamorphic relation 1: add typo ------------------")
    fail_test = metamorphic_tester.run_perturbation(p1, "INV")
    print("FAILING TESTS:", fail_test)

    print("---------------metamorphic relation 2: change name ------------------")
    metamorphic_tester.run_perturbation(p2, "INV")
    print("---------------metamorphic relation 3: change location ------------------")
    metamorphic_tester.run_perturbation(p3, "INV")
    print("---------------metamorphic relation 4: change numbers ------------------")
    metamorphic_tester.run_perturbation(p4, "INV")
    print("---------------metamorphic relation 5: add/remove punc ------------------")
    metamorphic_tester.run_perturbation(p5, "INV")
    print("---------------metamorphic relation 6: add negation ------------------")
    metamorphic_tester.run_perturbation(p6, "CHANGE")
    print("---------------metamorphic relation 7: add negation phrase------------------")
    metamorphic_tester.run_perturbation(p7, "MONO_DEC")

def run_experiment(model, cov_metrics, guided, label):
    # Read the data from the SST-2 dataset
    base = Path('SST-2')
    sents = base / 'datasetSentences.txt'
    split = base / 'datasetSplit.txt'
    df = pd.read_table(sents)
    df = df.join(pd.read_csv(split).set_index('sentence_index'), on='sentence_index')
    # Use the development set
    seeds = df[df['splitset_label']==2]['sentence'].head(n=200).values.tolist()
    metamorphic_tester = MetamorphicTester(model, cov_metrics, seeds)
    cov_incs, fail_test_list = metamorphic_tester.run_search(guided=guided)
    with Path('tmp/results.txt').open("a") as f:
        f.write("\n===========\n")
        f.write("Num of failures " + str(len(fail_test_list)) + '\n')
        f.write(json.dumps(fail_test_list, indent=4))
    return pd.DataFrame(dict(time=list(range(len(cov_incs))), value=cov_incs, Guide=label))

def draw_comparison(nsample=1, coverage_type=0):
    model = NeuralModel()

    coverage_fn = [DeepxploreCoverage, DeepgaugeCoverage, NearestNeighborCoverage][coverage_type]
    coverage_name = ['Deepxplore', 'Deepgauge', 'NearestNeighbor'][coverage_type]

    df = pd.DataFrame()
    for i in range(nsample):
        cov_metrics = coverage_fn()
        df = df.append(run_experiment(model, cov_metrics, True, coverage_name), ignore_index=True)
        cov_metrics = coverage_fn()
        df = df.append(run_experiment(model, cov_metrics, False, "None"), ignore_index=True)

    ax = sns.relplot(x="time", y="value", hue="Guide", kind="line", data=df)
    ax.set(xlabel='#Tests', ylabel='#Covered Neurons')
    plt.savefig('tmp/cov.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    draw_comparison(nsample=5)

    # model = NeuralModel()
    # cov_metrics = DeepxploreCoverage()

    # simple_cov_test(model, cov_metrics)
    # simple_perturb_test(metamorphic_tester)

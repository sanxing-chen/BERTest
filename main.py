import math
from pathlib import Path

import torch
from torch import nn
import numpy as np
from metamorphicTesting.generator import MetamorphicGenerator
from metamorphicTesting.tester import MetamorphicTester
import spacy
from pattern.en import sentiment
import matplotlib.pyplot as plt

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
    def __init__(self, threshold=1.0):
        self.cov_map = np.zeros(TEST_SPACE, dtype=np.bool)
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

def plot_experiment(model, guided, label):
    cov_metrics = DeepxploreCoverage()
    metamorphic_tester = MetamorphicTester(model, cov_metrics)
    cov_incs, fail_test_list = metamorphic_tester.run_search(guided=guided)
    plt.plot(cov_incs, label=label)

def draw_comparison(nsample=1):
    model = NeuralModel()

    for i in range(nsample):
        plot_experiment(model, True, "guided " + str(i))
        plot_experiment(model, False, "unguided " + str(i))

    plt.legend()
    plt.savefig('cov.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    draw_comparison(nsample=2)

    # model = NeuralModel()
    # cov_metrics = DeepxploreCoverage()

    # simple_cov_test(model, cov_metrics)
    # simple_perturb_test(metamorphic_tester)

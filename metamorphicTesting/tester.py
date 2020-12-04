import checklist
from checklist.perturb import Perturb
import numpy as np
import spacy
from pattern.en import sentiment
from metamorphicTesting.generator import MetamorphicGenerator
import random

class MetamorphicTester:
    def __init__(self, neur_model, cov_metrics):
        self.model = neur_model
        self.cov_metrics = cov_metrics
        self.seeds = ['This was a very nice movie directed by John Smith.',
               'Mary Keen was brilliant!',
               'I hated everything about New York.',
               'This movie was very bad!',
               'Jerry gave me 8 delicious apples.',
               'I really liked this movie.',
               'just bad.',
               'amazing.',
               ]
        self.nlp = spacy.load('en_core_web_sm')
        self.pseeds = list(self.nlp.pipe(self.seeds))
        
        self.plist = [
            MetamorphicGenerator.Perturb_add_typo,
            MetamorphicGenerator.Perturb_change_names,
            MetamorphicGenerator.Perturb_change_location,
            MetamorphicGenerator.Perturb_change_number,
            MetamorphicGenerator.Perturb_punctuation,
            # MetamorphicGenerator.Perturb_add_negation,
            MetamorphicGenerator.Perturb_add_negation_phrase
        ]
        self.pnum = len(self.plist)

    def get_perturbation(self, sent, pid):
        dataset = [sent]
        if pid not in [0, 5]:
            dataset = list(self.nlp.pipe(dataset))
        pert_sent = self.plist[pid](dataset)
        expectation = 'INV' if pid < 5 else 'MONO_DEC'
        return pert_sent, expectation
    
    def run_search(self, guided=False):
        cov_incs = []
        sents = self.seeds[:]

        for sent in sents:
            pred, output_state = self.model.run_example(sent)
            cov_inc = self.cov_metrics.get_cov_inc(output_state, True)
        init_cov = self.cov_metrics.get_current()
        cov_incs.append(init_cov)

        pqueue = []
        max_tries = 150
        num_failure = 0
        fail_test_list = []
        while(len(cov_incs) < max_tries):
            if len(sents) == 0:
                sents += self.seeds[:]
            if len(pqueue) > 0:
                pid = pqueue.pop(0)
            else:
                pid = random.randint(0, 5)
            sent = sents.pop(0)
            # print(sent, pid)
            perturb, expect_bahavior = self.get_perturbation(sent, pid)
            if len(perturb.data) == 0:
                continue
            fail_test, cov_inc = self.run_perturbation(perturb, expect_bahavior)
            if len(fail_test) > 0:
                fail_test_list += fail_test
                num_failure += 1
            cov_incs.append(cov_inc[1])
            assert cov_inc[0] == 0
            if guided:
                if cov_inc[1]:
                    sents.append(sent)
                    sents.append(perturb.data[0][1])
                    pqueue.append(pid)
            else:
                sents.append(sent)
                sents.append(perturb.data[0][1])
        for i in range(1, len(cov_incs)):
            cov_incs[i] += cov_incs[i - 1]
        print("num_failure", num_failure)
        return cov_incs, fail_test_list

    # return sentences pair that failed
    def run_perturbation(self, perturb, expect_bahavior):
        fail_test = []
        all_cov_inc = []
        for test_pair in perturb.data:
            result, cov_inc = self.oracle_test(test_pair, expect_bahavior)
            if result == 0:
                fail_test.append(test_pair)
            all_cov_inc.extend(cov_inc)
        return fail_test, all_cov_inc

    # Check if the test pass or fail based on expected behavior
    # return 0 if metamorphic test fail, 1 if success
    def oracle_test(self, test_pair, expect_bahavior):
        sentiment = []
        cov_inc = []

        # Ensure there is only one test pair
        test_pair = test_pair[:2]

        for sentence in test_pair:
            pred, output_state = self.model.run_example(sentence)
            cov_inc.append(self.cov_metrics.get_cov_inc(output_state, True))
            if pred[0][0] > pred[0][1]:
                sentiment.append(-1)  # negative
            elif pred[0][0] < pred[0][1]:
                sentiment.append(1)  # positive
            else:
                sentiment.append(0)  # neutral
        # print ("sentiment prediction",sentiment)

        if expect_bahavior == "INV":  # prediction should not change
            success = all(ele == sentiment[0] for ele in sentiment)
            result = 1 if success else 0

        elif expect_bahavior == "CHANGE":  # prediction should change
            # pass if original and negation sentence prediction change
            result = 1 if sentiment[0] != sentiment[1] else 0  # fail

        elif expect_bahavior == "MONO_DEC":  # prediction should change
            # p7 sentiment = sentiment of new,old pair: [new, old, new, old]
            # sentiment at even index i should not be less than i+1
            # pass if all pairs' is mono decreasing
            result = 0 if sentiment[0] < sentiment[1] else 1
        
        return result, cov_inc

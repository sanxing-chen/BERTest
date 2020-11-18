import checklist
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.pred_wrapper import PredictorWrapper
from checklist.expect import Expect
import numpy as np


class MetamorphicGenerator:
    def __init__(self):
        pass

    #invariant: add typo eg. amazing --> mmazing
    def Perturb_add_typo(self,dataset):
        return Perturb.perturb(dataset, Perturb.add_typos)

    def Perturb_change_names(self,pdataset):
        return Perturb.perturb(pdataset, Perturb.change_names)

    #invariant: change location eg. New York --> CVille
    def Perturb_change_location(self,pdataset):
        return Perturb.perturb(pdataset, Perturb.change_location)

    #invariant:change numbers eg. 1 --> 3
    def Perturb_change_number(self,pdataset):
        return Perturb.perturb(pdataset, Perturb.change_number)

    #invariant: add or remove punctuation. (might not be very useful though)
    def Perturb_punctuation(self,pdataset):
        return Perturb.perturb(pdataset, Perturb.punctuation)

    #add negation eg. like --> don't like 
    #prediction should change
    #negation function doesn't work on many sentences
    def Perturb_add_negation(self,pdataset):
        return Perturb.perturb(pdataset, Perturb.add_negation)

    #add negation phrase at the end of the sentence
    #prediction should mono decreasing
    def Perturb_add_negation_phrase(self,dataset):
        def add_negative_phrase(x):
            phrases = ['Anyway, I thought it was bad.', 'Having said this, I hated it', 'The director should be fired.']
            return ['%s %s' % (x, p) for p in phrases]
        return Perturb.perturb(dataset, add_negative_phrase)
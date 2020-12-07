from .checklist.test_suite import TestSuite
from .checklist.perturb import Perturb
from .checklist.test_types import MFT, INV, DIR
from .checklist.pred_wrapper import PredictorWrapper
from .checklist.expect import Expect
import numpy as np
import random

class MetamorphicGenerator:
    def __init__(self):
        pass

    # invariant: add typo eg. amazing --> mmazing
    @staticmethod
    def Perturb_add_typo(dataset):
        return Perturb.perturb(dataset, Perturb.add_typos)

    @staticmethod
    def Perturb_change_names(pdataset):
        return Perturb.perturb(pdataset, Perturb.change_names)

    # invariant: change location eg. New York --> CVille
    @staticmethod
    def Perturb_change_location(pdataset):
        return Perturb.perturb(pdataset, Perturb.change_location)

    # invariant:change numbers eg. 1 --> 3
    @staticmethod
    def Perturb_change_number(pdataset):
        return Perturb.perturb(pdataset, Perturb.change_number)

    # invariant: add or remove punctuation. (might not be very useful though)
    @staticmethod
    def Perturb_punctuation(pdataset):
        return Perturb.perturb(pdataset, Perturb.punctuation)

    # add negation eg. like --> don't like
    # prediction should change
    # negation function doesn't work on many sentences
    @staticmethod
    def Perturb_add_negation(pdataset):
        return Perturb.perturb(pdataset, Perturb.add_negation)

    # add negation phrase at the end of the sentence
    # prediction should mono decreasing
    @staticmethod
    def Perturb_add_negation_phrase(dataset):
        def add_negative_phrase(x):
            phrases = [ 'Having said this, I hated it', 'Anyway, I thought it was bad.', 'I do not like this', 'Bad!','Not good!']
            rand = random.randint(0, 4)
            return ['%s %s' % (x, phrases[rand])]
        return Perturb.perturb(dataset, add_negative_phrase)

    @staticmethod
    def Perturb_change_gender(pdataset):
        return Perturb.perturb(pdataset, Perturb.change_gender)

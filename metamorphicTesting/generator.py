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


    @staticmethod
    def Perturb_add_irrelevant_phrase(dataset):
        import string
        def random_string(n):
            return ''.join(np.random.choice([x for x in string.ascii_letters + string.digits], n))
        def random_url(n=6):
            return 'https://t.co/%s' % random_string(n)
        def random_handle(n=6):
            return '@%s' % random_string(n)

        # data['sentence']

        def add_irrelevant(sentence):
            # urls_and_handles = [random_url() for _ in range(5)] + [random_handle() for _ in range(5)]
            # irrelevant_before = ['@airline '] + urls_and_handles
            # irrelevant_after = urls_and_handles 
            # rets = ['%s %s' % (x, sentence) for x in irrelevant_before]
            # rets += ['%s %s' % (sentence, x) for x in irrelevant_after]
            my_irrelevant = ['OK.', 'I mean.', 'Yup.', 'Yeah.']
            rets = ['%s %s' % (x, sentence) for x in my_irrelevant]
            rand = random.randint(0, 3)
            return [rets[rand]]

        return Perturb.perturb(dataset, add_irrelevant)


import checklist
from checklist.perturb import Perturb
import numpy as np

class MetamorphicTester:
    def __init__(self, neur_model, cov_metrics):
        self.model = neur_model
        self.cov_metrics = cov_metrics

    def run_perburtation(self, perturb, expect_bahavior):
        for test_pair in perturb.data:
            print ("Pass(1) or fail(0): ", self.oracle_test(test_pair,expect_bahavior)) 

    #Check if the test pass or fail based on expected behavior
    # return 0 if metamorphic test fail, 1 if success
    def oracle_test(self, test_pair, expect_bahavior):
        sentiment = []
        print (test_pair)
        for sentence in test_pair:
            pred, output_state = self.model.run_example(sentence)
            cov_inc = self.cov_metrics.get_cov_inc(output_state)
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



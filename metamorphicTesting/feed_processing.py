from pathlib import Path
import pandas as pd
from checklist.perturb import Perturb
import spacy

filename = "contains1.txt"
base = Path('SST-2')
sents = base / 'datasetSentences.txt'
split = base / 'datasetSplit.txt'
df = pd.read_table(sents)
df = df.join(pd.read_csv(split).set_index('sentence_index'), on='sentence_index')
seeds = df[df['splitset_label']==2]['sentence'].values.tolist()

# only use sentence have label !=3
filter_seed = open(filename, "w", encoding='utf-8')
nlp = spacy.load('en_core_web_sm')

pdataset = list(nlp.pipe(seeds))
for i in range (0,len(pdataset)):
    trans1 = Perturb.change_names(pdataset[i])
    trans2 = Perturb.change_location(pdataset[i])
    trans3 = Perturb.change_number(pdataset[i])
    # if ((trans1!=None and trans3 != None)or (trans2 != None and trans3 != None)or (trans1!=None and trans2 != None)):
    if (trans3!=None or trans2 !=None or trans1 != None):
        filter_seed.write(seeds[i]) 
        filter_seed.write("\n")
filter_seed.close()


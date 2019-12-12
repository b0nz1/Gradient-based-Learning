# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
import codecs

def read_data(fname):
    data = []
    with codecs.open(fname,'r',encoding='utf8') as f:
        line = f.readline() 
        while line:
            label, text = line.strip().lower().split("\t",1)
            data.append((label, text))
            line = f.readline()
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]
    # switch to letter-unigram
    #return ["%s" %(c1) for c1 in zip(text)]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]
TEST  = [(l,text_to_bigrams(t)) for l,t in read_data("test")]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}


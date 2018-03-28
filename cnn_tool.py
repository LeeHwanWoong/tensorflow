import numpy as np
import pandas as pd
import re
import tensorflow as tf
import random


####################################################
# cut words function                               #
####################################################
def cut(contents, cut=2):
    results = []
    for content in contents:
        words = content.split()
        result = []
        for word in words:
            result.append(word[:cut])
        results.append(' '.join([token for token in result]))
    return results

####################################################
# divide train/test set function                   #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:int(round(train_prop * len(x)))]
    y_tr = y[tmp][:int(round(train_prop * len(x)))]
    x_te = x[tmp][-int((len(x)-round(train_prop * len(x)))):]
    y_te = y[tmp][-int((len(x)-round(train_prop * len(x)))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
def make_input(documents, max_document_length):

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(documents)))

    # Extract word:id mapping from the object.

    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# make output function                             #
####################################################
def make_output(points, threshold):
    results = np.zeros((len(points),2))
    for idx, point in enumerate(points):
        if point > threshold:
            results[idx,0] = 1
        else:
            results[idx,1] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path, eng=True, num=True, punc=False):
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents = []
    points = []
    for idx,doc in enumerate(corpus):
        if isNumber(doc[0]) is False:
            print type(doc[0]), doc[0]
            contents.append(doc[0])
            points.append(doc[1])
        if idx % 100000 is 0:
            print('%d docs / %d save' % (idx, len(contents)))
    print contents, points
    return contents, points

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False
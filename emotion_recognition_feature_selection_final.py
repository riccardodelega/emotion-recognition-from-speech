#!/usr/local/bin/python
# coding: utf-8

'''
Created on 16/set/2014

@author: Marco Ciccone, Riccardo Delegà
'''

from __future__ import division   # for floating-point division as a default
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk import ConfusionMatrix
from nltk.tag import hmm
from itertools import combinations
import os, sys, io
import time
import random
import numpy as np
from scipy import misc
import Orange
import Orange.feature
import Orange.data
import collections
import new
from math import ceil
from __builtin__ import raw_input

verbose = 0

#######################################################################################################################
### Functions to import the dataset
#######################################################################################################################

def import_corpus():
    
    '''
        In the corpus we have 6 categories representing the emotions.
        for each categories we have 180 examples (but "sadness" that we have 179)
        each example is a sentence
        each sentence has been tokenize, so for each word (token) we have extracted
        a set of features.
        
        the model structure of the corpus is :
         
        corpus[ category ] [ observation ] [ token ] [ feature ]
         
        # joy (gioia) - 180 files
        # neutral (neutro) - 180 files
        # fear (paura) - 180 files
        # anger (rabbia) - 180 files
        # sadness (tristezza) - 179 files 
    '''   
    
    POS = import_POS()
    
    
    path = "./features/"
    categories = ["joy","neutral","fear","anger","sadness"]
    
    corpus = {}
    for cat in categories :
        folder_path = path+cat+"/"
        observations = read_features_from_folder(folder_path,POS)
        
        corpus[cat] = observations
        #random shuffle of the observation to obtain different results
        random.shuffle(corpus[cat])
    return corpus

def read_features_from_file(filename):
    # read from file
    #print filename
    
    if ".txt" in filename : 
        #f = open(filename, encoding="utf-16")
        f = io.open(filename, 'r', encoding = 'utf-16be')
        lines = f.readlines()
        f.close()
        # first row is the header
        features = []
        header = lines.pop(0)
        for line in lines :
            lst = line.split('\t'); # split every feature into a list 
            word = lst.pop(0).encode('ascii','ignore') # take the word that the features are about 
            l = [float(i) for i in lst] #cast the values of the features from string to float
            l.insert(0,word) # create the list of the features
            l.append("")
            features.append(l)
        #return np.matrix(features)
        return features
    else : 
        return []

def read_features_from_folder(folder,POS):
    observations = []
    for filename in os.listdir(folder) :
        #print(filename)
        index_pos = filename[:3]
        o = read_features_from_file((folder+filename))
        if o :
            i=0
            for p in POS[index_pos]:
                o[i][len(o[i])-1] = p
                i+=1
            observations.append(o)
    return observations

def load_header():
    path = "./features/"
    categories = ["joy","neutral","fear","anger","sadness"]
    for cat in categories:
        folder = path+cat+"/"
        for filename in os.listdir(folder) :
            if ".txt" in filename : 
                f = io.open(folder+filename, 'r', encoding = 'utf-16be')
                lines = f.readlines()
                f.close()
                # first row is the header
                header = lines.pop(0)
                header = header.split('\t'); # split every feature into a list
                header[0] = "token" 
                return header
        else : 
            return []

def read_POS(filename):
    if ".txt" in filename:
        f = io.open(filename, 'r', encoding = 'utf-8')
        lines = f.readlines()
        f.close()
        POS_tags = []
        for line in lines:
            lst = line.split('\t')
            POS_tags.append(str(lst[1]))
        return POS_tags
    else:
        pass

def import_POS():
    folder = "./POS/"
    pos_collection = {}
    for filename in os.listdir(folder):
        pos = read_POS(folder+filename)
        index_pos = filename[:-4]
        pos_collection[index_pos] = pos
    return pos_collection
#######################################################################################################################
### Functions to normalize and discretize the features
#######################################################################################################################

def create_features_matrix(corpus,index_f):
    '''create a matrix (a list o list) that contains all the values for each feature aggregated from the categories'''
    features_matrix = [[] for x in index_f]
    for cat in corpus.keys() :
        for sentence in corpus[cat] :
            for token in sentence :
                for i in index_f :
                    features_matrix[i-1].append(token[i])
    return features_matrix

def find_min_max_features(features_matrix):
    '''extract the minimum and the maximum for each aggregated features to normalize them'''
    feature_max = []
    feature_min = []
    for feature in features_matrix :
        feature_max.append(max(feature))
        feature_min.append(min(feature))
    return feature_min,feature_max
    
def normalize_features(corpus,index_f =[]):
    if not index_f:
        index_f = range(0,10) # if not specified then normalize all features
    features_matrix = create_features_matrix(corpus,index_f)
    feature_min,feature_max = find_min_max_features(features_matrix)
    #for i in range(len(feature_max)):
    #    print feature_min[i],"   ",feature_max[i] 
    '''normalize the values of the features between 0 and 1 with the formula (x-xmin)/(xmax-xmin)'''
    for cat in corpus.keys() :
        for sentence in corpus[cat] :
            for token in sentence :
                for i in index_f :
                    token[i] = (token[i] - feature_min[i-1])/(feature_max[i-1]-feature_min[i-1])          

def createOrangeDataTable(values_f, name_f, index_f):
    #define orange.feature
    orange_f = Orange.feature.Continuous(name_f)
    #define orange.data.domain
    domain = Orange.data.Domain(orange_f,False)
    #convert corpus to orange.data.table
    d = np.array(values_f).reshape(len(values_f),1)            
    data_table = Orange.data.Table(domain,d)
    return data_table

def discretize_features(corpus, header, index_avarage_features, index_delta_features) :
    index_f = index_avarage_features + index_delta_features
    matrix_features = create_features_matrix(corpus,index_f)
    for i in index_f:
        if i in index_avarage_features:
            discretize_avarage_features(corpus,matrix_features[i-1],header[i],i)
        if i in index_delta_features:
            discretize_delta_features(corpus,matrix_features[i-1],header[i],i)                   

def discretize_avarage_features(corpus, values_f, name_f, index_f):            
    data_table = createOrangeDataTable(values_f, name_f, index_f)
    # orange discretization
    disc_simple_f = Orange.data.discretization.DiscretizeTable(data_table,
    method = Orange.feature.discretization.EqualFreq(n=3))
    # label discretization
    attrs_f = disc_simple_f.domain.attributes
    for cat in corpus.keys():
        for sentence in corpus[cat]:
            for word in sentence:
                word[index_f] = assignLabel_avarage(attrs_f,word[index_f])

def discretize_delta_features(corpus, values_f, name_f, index_f):
    data_table = createOrangeDataTable(values_f, name_f, index_f)
    # orange discretization
    disc_delta_f = Orange.data.discretization.DiscretizeTable(data_table,
    method = Orange.feature.discretization.EqualFreq(n=5))
    # label discretization
    attrs_f = disc_delta_f.domain.attributes
    for cat in corpus.keys():
        for sentence in corpus[cat]:
            for word in sentence:
                word[index_f] = assignLabel_delta(attrs_f,word[index_f])

def assignLabel_avarage(attrs_f,f_value):
    label = ''
    cut_points = [p for p in attrs_f[0].get_value_from.transformer.points]
    if f_value<= cut_points[0]:
        label = 'low'
    else:
        if f_value> cut_points[0] and f_value<= cut_points[1]:
            label = 'mid'
        else:
            if f_value> cut_points[1]:
                label = 'high'
    return label

def assignLabel_delta(attrs_d,d_value):
    label = ''
    cut_points = [p for p in attrs_d[0].get_value_from.transformer.points]
    if d_value<= cut_points[0]:
        label = '--slope'
    else:
        if d_value>cut_points[0] and d_value<=cut_points[1]:
            label = '-slope'
        else:
            if d_value>cut_points[1] and d_value<=cut_points[2]:
                label = 'flat'
            else:
                if d_value>cut_points[2] and d_value<=cut_points[3]:
                    label = '+slope'
                else:
                    if d_value>cut_points[3]:
                        label = '++slope'
    return label

#######################################################################################################################
### Functions to train the HMM and test the accuracy 
#######################################################################################################################

def split_corpus(corpus, train_set_fraction):    
    '''Splitting the corpus in training and dataset ''' 
    train_set = {}
    test_set = {}
    for cat in corpus.keys():
        train_set_limit = int(train_set_fraction * len(corpus[cat]))
        train_set[cat] = corpus[cat][:train_set_limit] 
        test_set[cat] = corpus[cat][train_set_limit:]               
    return train_set, test_set


def train(corpus, train_set, test_set, index_features):
    '''Training the models'''
    
    observations = []
    # extract all the features all the sentences of all the categories
    # !!FROM THE ENTIRE CORPUS NOT ONLY THE TRAINING SET!!
    for cat in corpus.keys():
        for sentence in corpus[cat]:
            for word in sentence:                
                '''observations.append(tuple(word))'''
                # feature subset selection
                new_word = []
                for feature in index_features : 
                    new_word.append(word[feature])
                observations.append(tuple(new_word))       
    '''
    raw_input()
    for o in observations : 
        print o
    '''
                
    # observation is a list of tuples because the HiddenMarkovModelTrainer of NLTK needs tuples instead of list
    # observation are needed by the hmm class for building the probability table
    symbols = list(set(observations)) # symbols must be a list of unique values of observation
    #print "symbols: ", len(symbols)
    #print "observations: ", len(observations)
    hmms = train_hmm(train_set, symbols,index_features)
    return test(hmms, test_set,index_features)

def train_hmm(train_set, observations, index_features):    
    '''Training the hmms...'''
    # symbols is a vector of inputs, each input is a vector of features (requires to be tuples by nltk)
    # we don't need to specify the states so we choose 1 and 2
    trainer = HiddenMarkovModelTrainer(states = [1,2], symbols = observations) 
    hmms = {}
    for cat in train_set.keys():
        print "Training HMM of cat:",cat
        tuple_sentences = []
        for sentence in train_set[cat]:
            
            '''tuple_sentence = [(tuple(word),'') for word in sentence]
            tuple_sentences.append(tuple_sentence)
            '''
            '''feature subset selection'''
            new_sentence = []
            for word in sentence :
                new_word = []
                for feature in index_features : 
                    new_word.append(word[feature])
                new_sentence.append(new_word)    
            tuple_sentence = [(tuple(word),'') for word in new_sentence]
            tuple_sentences.append(tuple_sentence)
            
            # sentence is a list of list! so w is a list of feature not only a word!
        hmms[cat] = trainer.train_unsupervised(tuple_sentences, max_iterations=10)
    return hmms

def test(hmms, test_set,index_features):
    total = 0
    correct = 0
    correct_cat = []
    predicted_cat = []
    for cat in test_set.keys():
        for sentence in test_set[cat]:
            '''test_sentence = [(tuple(word),'') for word in sentence]'''
            
            '''feature subset selection'''
            new_sentence = []
            for word in sentence :
                new_word = []
                for feature in index_features : 
                    new_word.append(word[feature])
                new_sentence.append(new_word)    
            test_sentence = [(tuple(word),'') for word in new_sentence]
            
            if verbose:
                pass
            
            max_prob = -1
            sentence_cat = random.choice(test_set.keys()) #assign a random category just to initialize
            
            # test the probabilities that the sentence is of a type of emotion
            # the higher probability is the winner
            #if verbose:
            #    print "Probabilities of each hmm : "
            for c in hmms.keys():
                sentence_prob = hmms[c].probability(test_sentence)
                #if verbose:
                #    print c," : ",sentence_prob
                if sentence_prob > max_prob:
                    sentence_cat = c
                    max_prob = sentence_prob
            #if verbose:
            #    print ""
            #print ""
            
            correct_cat.append(cat) # we save the correct category in a list that we'll use to build ConfusionMatrix
            predicted_cat.append(sentence_cat) # we save the category predicted in a list that we'll use to build ConfusionMatrix 
            if (cat == sentence_cat):
                correct += 1
            total += 1
    try:
        accuracy = ((correct / total)*100)
    except ZeroDivisionError:
        accuracy = 0 # error
    
    # the confusionMatrix function needs the list of the correct label and the list of the predicted
    matrix = ConfusionMatrix(correct_cat, predicted_cat)
    
    print "correct:", correct
    print "total:", total
    print "the accuracy is: %.2f%%" % accuracy
    print matrix
    return accuracy, matrix

#######################################################################################################################
### Feature selection algorithms and cross-validation
#######################################################################################################################

def forward_stepwise_feature_selection(corpus, train_set, test_set):
    index_features = range(1,12)
    features_set = []
    best_features_set = []
    while index_features:
        print "features_set : ",features_set
        print "index_features : ",index_features
        print "" 
        best_accuracy = -1
        for i in  index_features: 
            feature_temp = list(features_set)
            feature_temp.append(i)
            print "features set : " , feature_temp
            accuracy, matrix = train(corpus, train_set, test_set, feature_temp)
            if accuracy > best_accuracy:
                best_index = i
                best_accuracy = accuracy
        features_set.append(best_index)
        index_features.remove(best_index)
        best_features_set.append([list(features_set),best_accuracy])
    return best_features_set

def best_feature_selection(corpus, train_set, test_set) : 
    index_features = range(1,11)
    best_features_set = []
    accuracies = []
    for k in range(1,11):
        features_nk = list(combinations(index_features,k))
        best_accuracy = -1
        for features_temp in features_nk : 
            #print "features_set : ",features_temp
            #print "" 
            accuracy, matrix = train(corpus,train_set,test_set,features_temp)
            if accuracy > best_accuracy:
                best_comb = features_temp
                best_accuracy = accuracy
            accuracies.append([features_temp,accuracy])
        best_features_set.append([best_comb,best_accuracy])
    return accuracies,best_features_set

def k_fold_cross_validation(corpus, k=5):
    train_set_fraction = 1 - 1/k #e.g. if k = 5, train_set_function = 0.8
    sets = [{} for x in range(k)]
    
    for cat in corpus.keys():
        delim = int(ceil(len(corpus[cat]) / float(k)))
        for i in range(0,k):
            sets[i][cat] = corpus[cat][i*delim:delim*(i+1)]

    cross_train_set = []
    cross_test_set = []
    for j in range(0,k):
        train_set = {}
        for cat in corpus.keys():
            train_set[cat] = []
        test_set = {}
        for l in range(0,k):
            if i == l: #in test set
                test_set = sets[l]
                cross_test_set.append(test_set)
            else: # in train set
                for cat in corpus.keys():
                    for sentence in sets[l][cat]:
                        train_set[cat].append(sentence)
                cross_train_set.append(train_set)
    return cross_train_set,cross_test_set

def main():
    # import the audio and textual features
    corpus = import_corpus()
    ''' 
        0 token  : token of a word 
        1 avP    : avaragePitch
        2 dP     : deltaPitch
        3 avI    : avarageIntensity
        4 dI     : deltaIntensity
        5 f1     : avarageFormant1
        6 dF1    : deltaFormant1
        7 f2     : avarageFormant2
        8 dF2    : deltaFormant2 
        9 h      : harmonicity
        10 pr    : phoneRate
        11 pos   : pos tag
    '''              
    
    '''
        if I dont consider the pos tag : number of features 11
        if I consider the pos tag : number of features 11
    '''
    #header = load_header()
    header = ["token","avP","dP","avI","dI","f1","dF1","f2","dF2","h","pr","pos"]
    # normalize all the features
    
    normalize_features(corpus,range(1,11))
    #discretize the features
    index_avarage_features = [1,3,5,7,9,10]
    index_delta_features = [2,4,6,8]
    discretize_features(corpus,header,index_avarage_features,index_delta_features)
    
    # just check if the discrete class are uniformly distributed
    ''' 
    features_matrix = create_features_matrix(corpus, range(1,11))
    i = 1
    for f in features_matrix :
        counter = collections.Counter(f)
        print header[i]
        print(counter.most_common(5))
        print
        i=i+1
    '''
    
    # BESTSUBSET SELECTION (NO CROSS-VALIDATION
    '''
    accuracies,best_features_set = best_feature_selection(corpus, train_set, test_set)
    print accuracies
    print best_features_set
    '''
    
    # FORWARD STEPWISE SELECTION (WITH CROSS-VALIDATION
    '''
    k = 5
    #train_set_fraction = 1 - 1/k #e.g. if k = 5, train_set_function = 0.8
    cross_train_sets, cross_test_sets = k_fold_cross_validation(corpus, k) 
    
    accuracies = []
    for j in range(0,k):
        #accuracy = forward_stepwise_feature_selection(corpus, cross_train_sets[j], cross_test_sets[j])
        accuracy, matrix = train(corpus, cross_train_sets[j], cross_test_sets[j], range(1,12))
        accuracies.append(accuracy)
    print accuracies
        '''
    
    # NO CROSS-VALIDATION OR FEATURE SELECTION
    train_set, test_set = split_corpus(corpus, train_set_fraction = 0.8)
    accuracy, matrix = train(corpus, train_set, test_set, range(0,1))
    print accuracy
    accuracy, matrix = train(corpus, train_set, test_set, range(0,12))
    print accuracy
    accuracy, matrix = train(corpus, train_set, test_set, [1,5,8,7])
    print accuracy
    
    #accuracies = []
    #feature_set = [[0],[0,10],[0,1,10],[1, 3, 4, 5] ,[1, 3, 4, 7, 9] ,[1, 3, 5, 7, 9, 10] ,[1, 3, 6, 7, 8, 9, 10] ,[0, 1, 2, 4, 7, 8, 9, 10] ,[0, 1, 2, 3, 4, 6, 7, 9, 10] ,[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]
    #feature_set = [[1, 3, 5, 7, 9, 10]]
    '''
    for f in  feature_set:
        accuracy, matrix = train(corpus, train_set, test_set, f)
        accuracies.append(accuracy)
    print accuracies
    '''
    
    '''
    f = feature_set[0]
    # K-FOLD CROSS-CORRELATION
    k = 5
    train_set_fraction = 1 - 1/k #e.g. if k = 5, train_set_function = 0.8
    sets = [ {}, {}, {}, {}, {} ]
                
    for cat in corpus.keys():
        delim = int(ceil(len(corpus[cat]) / float(k)))
        for i in range(0,k):
            sets[i][cat] = corpus[cat][i*delim:delim*(i+1)]
                                
    for i in range(0,k):
        train_set = {}
        for cat in corpus.keys():
            train_set[cat] = []
        test_set = {}
        for j in range(0,k):
            if i == j: #in test set
                test_set = sets[i]
            else: # in train set
                for cat in corpus.keys():
                    for sentence in sets[j][cat]:
                        train_set[cat].append(sentence)
        # now you have the two sets
        accuracy, matrix = train(corpus, train_set, test_set, f)
        accuracies.append(accuracy)
    '''
    
if __name__ == '__main__':
    main()

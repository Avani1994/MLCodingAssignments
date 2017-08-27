import sys
import re
from pprint import pprint
import csv
from random import shuffle,random,uniform

def parseTraining(k, trainfile, trainlabel):
    trfile = open(trainfile, "r").read().splitlines()
    lfile = open(trainlabel, "r").read().splitlines()
    data = []
    labels = []
    finalinput = []
    for line in trfile:
        line = line.split(" ")
        data = data + [line]       

    for line in lfile:
        labels = labels + [line]

    for i,row in enumerate(data):
        row.extend([labels[i]])
        row.insert(0, 1)
    
    return data

def dot(x, w):
    if len(x) != len(w):
        return 0
    return sum(float(pair[0]) * float(pair[1]) for pair in zip(x, w))

def updateless1(x,w,c,gamma,n_trees):
    updated_wvec = []
    for pair in zip(x[:n_trees+1],w):
        wnew = (1-gamma)*float(pair[1]) + (gamma * c * float(x[-1]) * float(pair[0]))
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec

def updategreater1(w,gamma):
    updated_wvec = []
    for wt in w:
        wnew = (1-gamma)*wt 
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec    

def trainsvm(train_input,epochs,n_trees):
    gamma0 = 0.001
    gamma=gamma0
    t = 0.0
    c = 0.5
    wt_vec = [0]*(n_trees+1)
    #print wt_vec
    for epoch in range(1,epochs+1):
        shuffle(train_input)
        for example in train_input:
            result = dot(example[:n_trees+1],wt_vec)
            if(float(example[-1])*result <= 1):
                wt_vec = updateless1(example, wt_vec, c, gamma,n_trees+1)
            else:
                wt_vec = updategreater1(wt_vec, gamma)
            t = t + 1.0
            gamma = gamma0 / ( 1 + gamma0 * (t / c))
    return wt_vec

def accuracy(w, input,n_trees):
    error = 0.0
    total = 0.0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    p = 0.0
    r = 0.0
    f = 0.0
    for example in input:
        result = dot(example[:n_trees+1],w)
        if(result > 0.0 and float(example[-1]) > 0.0):
            true_positive = true_positive + 1.0
        if(result > 0.0 and float(example[-1]) < 0.0):
            false_positive = false_positive + 1.0
        if(result < 0.0 and float(example[-1]) > 0.0):
            false_negative = false_negative + 1.0
        if(float(example[-1])*result < 0.0):
            error = error + 1.0
        total = total + 1.0
    if(true_positive != 0.0 or false_positive != 0.0):
        p = true_positive / (true_positive + false_positive)
    if(true_positive != 0.0 or false_positive != 0.0):
        r = true_positive / (true_positive + false_negative)
    if(p != 0.0 or r != 0.0):
        f = 2.0 * p * r/(p + r)
    return {'accur': (total - error) / total, 'prec': p, 'recall': r, 'f_score': f}


def main():
    global types
    if (len(sys.argv) < 2):
        ''' 404 function not found '''
    else:

        trainfile = sys.argv[1]
        trainlabel = sys.argv[2]
        testfile = sys.argv[3]
        testlabel = sys.argv[4]
        train_input = parseTraining(1, trainfile, trainlabel)
        '''final_wt = trainsvm(train_input,1)
        Resulttrain = accuracy(final_wt, train_input)
        test_input = parseTraining(1,testfile,testlabel)
        Resulttest = accuracy(final_wt,test_input)
        accuracy_train = Resulttrain['accur']
        precision_train = Resulttrain['prec']
        recall_train = Resulttrain['recall']
        fscore_train = Resulttrain['f_score']
        accuracy_test = Resulttest['accur']
        precision_test = Resulttest['prec']
        recall_test = Resulttest['recall']
        fscore_test = Resulttest['f_score']
        #print(accuracy_train,"train accuracy")
        #print(precision_train,"train precision")
        #print(recall_train,"train recall")
        #print(fscore_train,"train fscore")
        #print(accuracy_test,"test accuracy")
        #print(precision_test,"test precision")
        #print(recall_test,"test recall")
        #print(fscore_test,"test fscore")
        '''

        
        
main()
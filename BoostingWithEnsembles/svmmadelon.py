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
        line = line.rstrip()
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

def updateless1(x,w,c,gamma):
    #print c, "c"
    #print type(c)
    #print gamma, "gamma"
    #print float(x[-1])
    updated_wvec = []
    for pair in zip(x[:501],w):
        wnew = (1-gamma)*float(pair[1]) + (gamma * c * float(x[-1]) * float(pair[0]))
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec

def updategreater1(w,gamma):
    updated_wvec = []
    for wt in w:
        wnew = (1-gamma)*wt 
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec    

def trainsvm(train_input,epochs,gamma0,c):
    #print train_input
    #print c,'c'
    #gamma0 = 0.01
    gamma=gamma0
    t = 0.0
    #c = 1.0
    wt_vec = [0]*501
    for epoch in range(1,epochs+1):
        shuffle(train_input)
        for example in train_input:
            result = dot(example[:501],wt_vec)
            #print example[-1], "example"
            #break
            if(float(example[-1])*result <= 1):
                wt_vec = updateless1(example, wt_vec, c, gamma)
            else:
                wt_vec = updategreater1(wt_vec, gamma)
            t = t + 1.0
            gamma = gamma0 / ( 1 + gamma0 * (t / c))
    return wt_vec

def accuracy(w, input):
    error = 0.0
    total = 0.0
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    p = 0.0
    r = 0.0
    f = 0.0
    for example in input:
        result = dot(example[:501],w)
        if(result > 0.0 and float(example[-1]) > 0.0):
            true_positive = true_positive + 1.0
        if(result > 0.0 and float(example[-1]) < 0.0):
            false_positive = false_positive + 1.0
        if(result < 0.0 and float(example[-1]) > 0.0):
            false_negative = false_negative + 1.0
        if(float(example[-1])*result < 0.0):
            error = error + 1.0
        total = total + 1.0
    #print true_positive
    #print false_positive
    #print false_negative
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
        test_input = parseTraining(1, testfile, testlabel)
        #print train_input
        shuffle(train_input)

        cv1 = train_input[:400]
        cv2 = train_input[400:800]
        cv3 = train_input[800:1200]
        cv4 = train_input[1200:1600]
        cv5 = train_input[1600:2000]
        #print cv_data
        cv_data = [cv1,cv2,cv3,cv4,cv5]
        
        #gamma0 = [0.1,0.05,0.001]
        gamma0 = [0.1,0.05,0.01,0.005,0.001]
        #c = [2.0,0.03125]
        c = [2.0,0.5,0.25,0.125,0.0625,0.03125]
        e = 2
        accu = {}
        Avg_accu = {}
        Weight = {}
        for cs in c:
            accu[cs] = {}
            Avg_accu[cs] = {}
            Weight[cs] = {}
            for g in gamma0:
                accu[cs][g] = {}
                Weight[cs][g] = {}
                sum_accu = 0.0
                for i in range(1,6):
                    data = []
                    for k in range(1,6):
                        if(k!=i):
                            data = data + cv_data[k-1]
                    final_wt = trainsvm(data,e,g,cs)
                    Weight[cs][g][i] = final_wt
                    Resultcv = accuracy(final_wt, cv_data[i-1])
                    accu[cs][g][i] = Resultcv['accur']
                    sum_accu = sum_accu + accu[cs][g][i]
                    #print accu[cs][g][i]
                Avg = sum_accu / 5.0    
                Avg_accu[cs][g] = Avg   
                #print data[0]
        print("Qn3.1.2 and Qn 3.1.3(precision, recall)")
        print("CROSS VALIDATION")
        pprint(Avg_accu)

        keys = max(((cs, g, i) for cs in accu for g in accu[cs] for epochs in accu[cs][g]), key=lambda (cs,g,i):accu[cs][g][i])
        print keys
        #print Weight[keys[0]][keys[1]][keys[2]]
        Resulttrain = accuracy(Weight[keys[0]][keys[1]][keys[2]],train_input)
        accuracy_train = Resulttrain['accur']
        precision_train = Resulttrain['prec']
        recall_train = Resulttrain['recall']
        fscore_train = Resulttrain['f_score']
        Resulttest = accuracy(Weight[keys[0]][keys[1]][keys[2]],test_input)
        accuracy_test = Resulttest['accur']
        precision_test = Resulttest['prec']
        recall_test = Resulttest['recall']
        fscore_test = Resulttest['f_score']
        print("TRAINING")
        print(accuracy_train,"train accuracy")
        print(precision_train,"train precision")
        print(recall_train,"train recall")
        print(fscore_train,"train fscore")
        print("TESTING")
        print(accuracy_test,"test accuracy")
        print(precision_test,"test precision")
        print(recall_test,"test recall")
        print(fscore_test,"test fscore")
        

        '''
        final_wt = trainsvm(train_input,1)
        print accuracy(final_wt, train_input)
        test_input = parseTraining(1,testfile,testlabel)
        print accuracy(final_wt, test_input)
        '''
main()
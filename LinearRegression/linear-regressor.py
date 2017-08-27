import sys
import re
from pprint import pprint
import csv
from random import shuffle,random,uniform
import math
import pylab

''' k could be used to read every second or third or .. line '''
# PARSE TRAINING and TEST SET
def parseTraininga2a(k, name):
    trfile = open(name, "r").read().splitlines()
    data = []
    for line in trfile:
        lines = []
        line = line.rstrip(' ')
        line = line.split(' ')
        for column in line:
            column = column.split(':')
            lines = lines + [column]
            #print lines 
        data = data + [lines]       
    return data

# CREATE 
def create_inputvector(data):
    #print("length of data is %s" %(len(data)))
    #global ques
    input_set =[]
    for row in data:
        x = []
        data_column = 0
        inp_col = 0
        #print("length of row is %s" %(len(row)))
        #print("1st row is %s" %(row))
        
        while(len(x)!= 125):
            if(data_column <= len(row)-1):
                #print("length of data_column is %s " %(data_column))
                if(row[data_column][0] == '+1' or row[data_column][0] == '-1'):
                    x = x + [int(row[data_column][0])]
                    x = x + [1]
                    data_column += 1
                else:
                    if(data_column == 1):
                        for i in range(int(row[data_column][0]) - 1):
                            x = x = x + [0]
                        x = x + [1]
                    else:
                        for i in range(int(row[data_column][0]) - int(row[data_column - 1][0]) - 1):
                            x = x + [0]
                        x = x + [1]
                    data_column += 1
            else:
                x = x + [0]
        input_set =input_set + [x]
    return input_set

def getval(x,w,y):
    b = prod3(y,w,x)
    c = math.log(b)
    return c

def Linear_regression(input_set, epochs, w,r,sigma,flag):
    #print w
    gamma=r
    t = 0.0
    c = 1.0
    sig = sigma
    lists = []
    epochss = []
    for epoch in range(1,epochs+1):
        shuffle(input_set)
        for example in input_set:
            #result = dot(example[1:],w[1:])
            
            w = update(example[1:], w[1:],example[0], gamma,sig)
            
            w.insert(0,0)
            
            t = t + 1.0
            gamma = r / ( 1 + r * (t / c))

        if(flag == 1):
            neg_log = 0
            for example in input_set:
                neg_log = neg_log + getval(example[1:],w[1:],example[0])
            neg_log = neg_log + div3(w[1:],sig)
            lists = lists + [neg_log]
            epochss = epochss + [epoch]
    return {'wt_vec':w, 'list':lists, 'epoc':epochss}
    
def accuracy(w, inputt):
    error = 0.0
    total = 0.0
    #true_positive = 0.0
    #false_positive = 0.0
    #false_negative = 0.0
    p = 0.0
    r = 0.0
    f = 0.0
    for example in inputt:
        #print example
        #print w
        result = dot(example[1:],w[1:])
        #print result
        '''
        if(result > 0.0 and float(example[-1]) > 0.0):
            true_positive = true_positive + 1.0
        if(result > 0.0 and float(example[-1]) < 0.0):
            false_positive = false_positive + 1.0
        if(result < 0.0 and float(example[-1]) > 0.0):
            false_negative = false_negative + 1.0
        '''
        if(example[0]*result < 0.0):
            error = error + 1.0
        total = total + 1.0
    #print total
    #print error
    '''
    if(true_positive != 0.0 or false_positive != 0.0):
        p = true_positive / (true_positive + false_positive)
    if(true_positive != 0.0 or false_positive != 0.0):
        r = true_positive / (true_positive + false_negative)
    if(p != 0.0 or r != 0.0):
        f = 2.0 * p * r/(p + r)
    '''
    return {'accur': (total - error) / total, 'prec': p, 'recall': r, 'f_score': f}

def div3(w,sig):
    summ = 0.0
    for wt in w:
        summ = summ + float(wt) * float(wt)
    final = summ / sig**2
    return final

def div1(w,sig):
    vec3 = []
    for wt in w:
        newit = (2.0*float(wt))/float(sig**2)
        vec3 = vec3 + [newit]
    return vec3

def prod1(x,label):
    vec1 = []
    for xi in x:
        term = -(float(xi) * label)
        vec1 = vec1 + [term]
    return vec1

def prod2(label,w,x):
    #print w
    #print x
    prod = 0.0
    for pair in zip(w,x):
        prod = prod + float(pair[0])*float(pair[1])
    #print prod
    prodf = prod*label
    final = 1 + math.exp(prodf)
    #print final
    return final

def prod3(label,w,x):
    #print w
    #print x
    prod = 0.0
    for pair in zip(w,x):
        prod = prod + float(pair[0])*float(pair[1])
    #print prod
    prodf = prod*label
    final = 1 + math.exp(-prodf)
    #print final
    return final

def div2(a,b):
    vec2 = []
    for item in a:
        newi = float(item)/float(b)
        vec2 = vec2 +[newi]
    return vec2


def grad(d,c):
    vec4 = []
    for pair in zip(d,c):
        item = float(pair[0]) + float(pair[1])
        vec4 = vec4 + [item]
    #print vec4
    return vec4



def update(x,w,label,gamma,sig):
    updated_wvec = []
    a = prod1(x,label)
    b = prod2(label,w,x)
    c = div1(w,sig)
    d = div2(a,b)
    gradient = grad(d,c)
    for pair in zip(gradient,w):
        wnew = pair[1] - (gamma * float(pair[0]))
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec

def wt_zeroes(number):
    return [0] * number

def dot(x, w):
    if len(x) != len(w):
        return 0
    return sum(pair[0] * pair[1] for pair in zip(x, w))                

def main():
    w = []
    '''as of now declare r as 1 change for different questions using commandline'''
    global ques
    if (len(sys.argv) < 1):
        ''' 404 function not found '''
    else:
        trainfile = sys.argv[1]
        #epochs = int(sys.argv[2])
        #print("epochs = %s" %(epochs))
        #decide_wts = int(sys.argv[2])
        #print("decide_wts = %s" %(decide_wts))
        #ques = int(sys.argv[3])
        #print("ques = %s" %(ques))
        testfile = sys.argv[2]
        #global test
        #test = 0
        
        #learning_rates = [1,0.1]
        epochss = [1]
        Accuracy = {}
        Error = {}
        Simple_Accuracy = {}
        Simple_Error = {}
        Weight = {}
        Simple_Weight = {}
        Avg_accu = {}
        flag = 0
        '''
        for i in range(125):
            w = w + [uniform(0,1)]
        '''
        trainingset = parseTraininga2a(1, trainfile)
        inputset = create_inputvector(trainingset)
        #print(len(inputset)), "...................................."
        cv1 = inputset[:1283]
        cv2 = inputset[1283:2566]
        cv3 = inputset[2566:3849]
        cv4 = inputset[3849:5132]
        cv5 = inputset[5132:6415]
        cv_data = [cv1,cv2,cv3,cv4,cv5]
        sigma = [1,5,10,100,1000,10000]
        r = 0.1
        epochs = 5
        for sig in sigma:
            Simple_Accuracy[sig] = {}
            #Simple_Error[r] = {}
            Simple_Weight[sig] = {}
            Avg_accu[sig] = {}
            sum_accu = 0.0
            for i in range(1,6):
                Simple_Accuracy[sig][i] = {}
                #Simple_Error[r] = {}
                Simple_Weight[sig][i] = {}
                for j in range(125):
                    w = w + [0]
                data = []
                for k in range(1,6):
                    if(k!=i):
                        data = data + cv_data[k-1]
                Finalized_wt_vec = Linear_regression(data,epochs,w,r,sig,flag)['wt_vec']
                Simple_Weight[sig][i] = Finalized_wt_vec
                Resultcv = accuracy(Finalized_wt_vec, cv_data[i-1])
                Simple_Accuracy[sig][i] = Resultcv['accur']
                sum_accu = sum_accu + Simple_Accuracy[sig][i]
            Avg_accu[sig] = sum_accu / 5.0
        print("-------------------------Cross Validation Results----------------------------")
        print("Average Accuracy of CV results for different values of sigma")
        pprint(Avg_accu)
        print("Accuracy for every split")
        pprint(Simple_Accuracy)   
        keys = max(((sig,i) for sig in Simple_Accuracy for i in Simple_Accuracy[sig]), key=lambda (sig,i):Simple_Accuracy[sig][i])
        print("Best value of Sigma")
        print keys[0]
        key_avg = max((sig for sig in Avg_accu),key=lambda (sig):Avg_accu[sig])
        testset = parseTraininga2a(1, testfile)
        input_testset = create_inputvector(testset)
        #print input_testset,"***************************************************88"
        #Simple_Weight[keys[0]][keys[1]].insert(0,uniform(0,1))
        print("-------------------LOGISTIC REGRESSION RESULTS -------------------------------")
        #print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
        #print Simple_Weight[keys[0]][keys[1]], "#######################"
        flag = 1
        epochs = 20
        Result =Linear_regression(inputset,epochs,w,r,sig,flag)
        listsss = Result['list']
        epochsss = Result['epoc']
        pylab.plot(epochsss,listsss)
        pylab.show()
        print("FINAL RESULTS ON TEST AND TRAIN SETS")
        Test_Result = accuracy(Simple_Weight[keys[0]][keys[1]],input_testset)
        print("Test Result Accuarcy: \n %s" %(Test_Result['accur']))
        Train_Result = accuracy(Simple_Weight[keys[0]][keys[1]],inputset)
        print("Training Result Accuracy : \n %s" %(Train_Result['accur']))
        #pylab.plot(epochss,)
        #test = 0
main()
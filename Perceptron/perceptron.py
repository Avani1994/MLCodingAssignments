import sys
import re
from pprint import pprint
import csv
from random import shuffle,random,uniform

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

# PARSE TRAINING SET FOR QN A
def parseQn1(k, name):
    trfile = open(name, "r").read().splitlines()
    data = []
    for line in trfile:
        lines = []
        line = line.rstrip(' ')
        line = line.split(' ')
        for column in line:
            column = column.split(':')
            lines = lines + [column] 
        data = data + [lines]        
    return data

# CREATE INPUT VECTOR X
def create_inputvector(data):
    #print("length of data is %s" %(len(data)))
    global ques
    input_set =[]
    for row in data:
        x = []
        data_column = 0
        inp_col = 0
        #print("length of row is %s" %(len(row)))
        #print("1st row is %s" %(row))
        if(ques == 1):
            while(len(x)!= 6):
                if(row[data_column][0] == '+1' or row[data_column][0] == '-1'):
                    x = x + [int(row[data_column][0])]
                else:
                    x = x + [int(row[data_column][1])]
                data_column += 1
            input_set =input_set + [x]
        
        elif(ques == 2 or ques == 3 or ques == 4):
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
# SIMPLE PERCEPTRON ALGORITHM    
def perceptron(input_set, epochs, w,r):
    global ques
    #print len(input_set)
    count = 0.0
    error = 0.0
    for epoch in range(epochs):
        if(ques == 2 or ques == 3):
            shuffle(input_set)
        for x in input_set:
            result = dot(x[1:],w[1:])
            #print("result = %s" %(result))
            if((x[0] * result) <= 0):
                error += 1.0
                if(test == 0):
                    w = update(x[1:],w[1:],r,x[0])
                    w.insert(0,uniform(0,1))
                #print("updated vector is : %s", w)
                #print("length of updated w is %s" %(len(w)))
            count += 1.0
    #print("Accuracy is %s for learning rate %s and epochs %s" %(accuracy(error, count), r, epochs))
    #print("Errors made by algo is %s for learning rate %s and epochs %s" %(r, error, epochs))
        #print("Final weight vector is %s for epoch %s" %(w, epoch))
    return {'Final weight vector': w[1:], 'accuracy' : accuracy(error, count), 'Errors' : error }

# MARGIN PERCEPTRON ALGORITHM
def Margin_perceptron(input_set, epochs, w,r,mu):
    global ques
    #print len(input_set)
    count = 0.0
    error = 0.0
    global test 
    for epoch in range(epochs):
        if(ques == 2 or ques == 3):
            shuffle(input_set)
        for x in input_set:
            result = dot(x[1:],w[1:])
            #print("result = %s" %(result))
            if((x[0] * result) <= mu):
                error += 1.0
                if(test == 0):
                    w = update(x[1:],w[1:],r,x[0])
                    w.insert(0,uniform(0,1))
                #print("updated vector is : %s", w)
                #print("length of updated w is %s" %(len(w)))
            count += 1.0
    #print("Accuracy is %s for learning rate %s and mu %s and epochs %s" %(accuracy(error, count), r, mu, epochs))
    #print("Errors made by algo is %s  for learning rate %s and mu %s and epochs %s" %(error, r, mu, epochs))
        #print("Final weight vector is %s for epoch %s" %(w, epoch))
    return {'Final weight vector': w[1:], 'accuracy' : accuracy(error, count), 'Errors' : error }

#AGGRESSIVE PERCEPTRON ALGORITHM
def Aggressive_perceptron(input_set, epochs, w, mu):
    global ques
    count = 0.0
    error = 0.0
    for epoch in range(epochs):
        if(flag == 1):
            shuffle(input_set)
        for x in input_set:
            result = dot(x[1:],w[1:])
            xintox = dot(x[1:],x[1:])
            #print("result = %s" %(result))
            if((x[0] * result) <= mu):
                error += 1.0
                if(test == 0):
                    eta = (mu - (x[0] * result)) / (xintox + 1)
                    w = update(x[1:],w[1:],eta,x[0])
                    w.insert(0,uniform(0,1))
                #print("updated vector is : %s", w)
                #print("length of updated w is %s" %(len(w)))
            count += 1.0
    #print("Accuracy is %s for max depth %s and epochs = %s" %(accuracy(error, count), mu, epochs))
    #print("Errors made by algo is %s for max depth %s and epochs = %s" %(error, mu, epochs))
        #print("Final weight vector is %s for epoch %s" %(w, epoch))
    return {'Final weight vector': w[1:], 'accuracy' : accuracy(error, count), 'Errors' : error}

#FINDS ACCURACY
def accuracy(error,count):
    return (count - error)/(count)

#UPDATES ON MISTAKE
def update(x,w,r,label):
    updated_wvec = []
    for pair in zip(x,w):
        wnew = pair[1] + (r * label * pair[0])
        updated_wvec = updated_wvec + [wnew]
    return updated_wvec

# CREATE WEIGHT VECTOR OF 0s FOR QN 1
def wt_zeroes(number):
    return [0] * number

# COMPUTE DOT PRODUCT OF X and W or X and X
def dot(x, w):
    if len(x) != len(w):
        return 0
    return sum(pair[0] * pair[1] for pair in zip(x, w))                

def main():
    w = []
    global ques
    if (len(sys.argv) < 1):
        ''' 404 function not found '''
    else:
        trainfile = sys.argv[1]
        decide_wts = int(sys.argv[2])
        print("decide_wts = %s" %(decide_wts))
        ques = int(sys.argv[3])
        print("ques = %s" %(ques))
        testfile = sys.argv[4]
        global test
        test = 0
        # QUESTION 1
        if(decide_wts == 0 and ques == 1):
            r = 1
            w = wt_zeroes(6)
            epochs = 1
            #print w
            trainingset = parseQn1(1, trainfile)
            inputset = create_inputvector(trainingset)
            Result = perceptron(inputset, epochs, w, r)
            Accuracy = Result['accuracy']
            Error    = Result['Errors']
            Final_Weight_Vector = Result['Final weight vector']
            print("Answers for question 1")
            print("Accuracy = %s" %(Accuracy))
            print("error = %s" %(Error))
            print("Final wt vector = %s" %(Final_Weight_Vector))
            #print trainingset
        # QUESTION 2
        elif(decide_wts == 1 and ques == 2):
            learning_rates = [1,0.1,0.01,0.001,0.0001,0.00001]
            epochss = [1]
            mus = [1,0.5,3,0.0001,5]
            Accuracy = {}
            Error = {}
            Simple_Accuracy = {}
            Simple_Error = {}
            Weight = {}
            Simple_Weight = {}
            '''
            for i in range(125):
                w = w + [uniform(0,1)]
            '''
            trainingset = parseTraininga2a(1, trainfile)
            inputset = create_inputvector(trainingset)
            for r in learning_rates:
                Simple_Accuracy[r] = {}
                Simple_Error[r] = {}
                Simple_Weight[r] = {}
                for epochs in epochss:
                    for i in range(125):
                        w = w + [uniform(0,1)]
                    Simple_Result = perceptron(inputset, epochs, w, r)
                    Simple_Accuracy[r][epochs] = Simple_Result['accuracy']
                    Simple_Error[r][epochs] = Simple_Result['Errors']
                    Simple_Weight[r][epochs] = Simple_Result['Final weight vector']
            print("-------------------SIMPLE PERCEPTRON TRAINING QN 2-------------------------------")
            print("Accuracy for Simple perceptron while training in Qn 2 with 1st key as learning rate, 2nd key as epochs and value is accuracy value")
            pprint(Simple_Accuracy)
            print("Errors for Simple perceptron while training in Qn 2 with 1st key as learning rate, 2nd key as epochs value is error value")
            pprint(Simple_Error)
            keys = max(((r, epochs) for r in Simple_Accuracy for epochs in Simple_Accuracy[r]), key=lambda (r,epochs):Simple_Accuracy[r][epochs])
            print("Best values are: (best learning  rate, best epoch)")
            print keys
            testset = parseTraininga2a(1, testfile)
            input_testset = create_inputvector(testset)
            test = 1
            Simple_Weight[keys[0]][keys[1]].insert(0,uniform(0,1))
            print("-------------------SIMPLE PERCEPTRON RESULTS QN 2-------------------------------")
            print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
            Test_Result = perceptron(input_testset, keys[1], Simple_Weight[keys[0]][keys[1]], keys[0])
            print("Test Result : \n %s" %(Test_Result))
            Train_Result = perceptron(inputset, keys[1], Simple_Weight[keys[0]][keys[1]], keys[0])
            print("Training Result : \n %s" %(Train_Result))
            test = 0

            for r in learning_rates:
                Accuracy[r] = {}
                Error[r] = {}
                Weight[r] = {}
                for mu in mus:
                    Accuracy[r][mu] = {}
                    Error[r][mu] = {}
                    Weight[r][mu] = {}
                    for epochs in epochss:
                        for i in range(125):
                            w = w + [uniform(0,1)]
                        Result = Margin_perceptron(inputset, epochs, w, r, mu)
                        Accuracy[r][mu][epochs] = Result['accuracy']
                        Error[r][mu][epochs]= Result['Errors']
                        Weight[r][mu][epochs] = Result['Final weight vector']
            print("-------------------MARGIN PERCEPTRON TRAINING QN2-------------------------------")
            print("Accuracy for Margine perceptron while training in Qn 2 with 1st key as learning rate, 2nd key as margin, 3rd key as epochs and value is accuracy value")
            pprint(Accuracy)
            print("Errors for Margine perceptron while training in Qn 2 with 1st key as learning rate, 2nd key as margin, 3rd key as epochs and value is error value")
            pprint(Error)
            keys = max(((r, mu, epochs) for r in Accuracy for mu in Accuracy[r] for epochs in Accuracy[r][mu]), key=lambda (r,mu,epochs):Accuracy[r][mu][epochs])            
            print("Best values are: (best learning rate, best margin, best epoch)")
            print(keys)
            test = 1
            Weight[keys[0]][keys[1]][keys[2]].insert(0,uniform(0,1))
            print("-------------------MARGIN PERCEPTRON RESULTS QN2-------------------------------")
            print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
            Margine_Test_Result = Margin_perceptron(input_testset, keys[2], Weight[keys[0]][keys[1]][keys[2]], keys[0], keys[1])
            print("Test Result : \n %s" %(Margine_Test_Result))
            Margine_Training_Result = Margin_perceptron(inputset, keys[2], Weight[keys[0]][keys[1]][keys[2]], keys[0], keys[1])
            print("Training Result : \n %s" %(Margine_Training_Result))
            test = 0

        # QUESTION 3
        elif(decide_wts == 1 and ques == 3):
            learning_rates = [1,0.1,0.01,0.001,0.0001,0.00001]
            mus = [1,0.1,0.001,4,5]
            epochss = [3,5]
            Accuracy = {}
            Error = {}
            Simple_Accuracy = {}
            Simple_Error = {}
            Weight = {}
            Simple_Weight = {}
            '''
            for i in range(125):
                w = w + [uniform(0,1)]
            '''
            trainingset = parseTraininga2a(1, trainfile)
            inputset = create_inputvector(trainingset)
            for r in learning_rates:
                Simple_Accuracy[r] = {}
                Simple_Error[r] = {}
                Simple_Weight[r] = {}
                for epochs in epochss:
                    for i in range(125):
                        w = w + [uniform(0,1)]
                    Simple_Result = perceptron(inputset, epochs, w, r)
                    Simple_Accuracy[r][epochs] = Simple_Result['accuracy']
                    Simple_Error[r][epochs] = Simple_Result['Errors']
                    Simple_Weight[r][epochs] = Simple_Result['Final weight vector']
            print("------------------- SIMPLE PERCEPTRON TRAINING QN3-------------------------------")
            print("Accuracy for Simple perceptron while training in Qn 3 with 1st key as learning rate, 2nd key as epochs and value is accuracy value")
            pprint(Simple_Accuracy)
            print("Errors for Simple perceptron while training in Qn 3 with 1st key as learning rate, 2nd key as epochs and value is error value")
            pprint(Simple_Error)
            keys = max(((r, epochs) for r in Simple_Accuracy for epochs in Simple_Accuracy[r]), key=lambda (r,epochs):Simple_Accuracy[r][epochs])
            print("Best values are: (best learning rate, best epoch)")
            print(keys)
            testset = parseTraininga2a(1, testfile)
            input_testset = create_inputvector(testset)
            test = 1
            Simple_Weight[keys[0]][keys[1]].insert(0,uniform(0,1))
            print("------------------- SIMPLE PERCEPTRON RESULTS QN3-------------------------------")
            print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
            Test_Result = perceptron(input_testset, keys[1], Simple_Weight[keys[0]][keys[1]], keys[0])
            print("Test Result : \n %s" %(Test_Result))
            Train_Result = perceptron(inputset, keys[1], Simple_Weight[keys[0]][keys[1]], keys[0])
            print("Training Result : \n %s" %(Train_Result))
            test = 0
            for r in learning_rates:
                Accuracy[r] = {}
                Error[r] = {}
                Weight[r] = {}
                for mu in mus:
                    Accuracy[r][mu] = {}
                    Error[r][mu] = {}
                    Weight[r][mu] = {}
                    for epochs in epochss:
                        for i in range(125):
                            w = w + [uniform(0,1)]
                        Result = Margin_perceptron(inputset, epochs, w, r, mu)
                        Accuracy[r][mu][epochs] = Result['accuracy']
                        Error[r][mu][epochs] = Result['Errors']
                        Weight[r][mu][epochs] = Result['Final weight vector']
            print("-------------------MARGIN PERCEPTRON TRAINING QN 3-------------------------------")
            print("Accuracy for Margine perceptron while training in Qn 3 with 1st key as learning rate, 2nd key as margin, 3rd key as epochs and value is accuracy value")
            pprint(Accuracy)
            print("Errors for Margine perceptron while training in Qn 3 with 1st key as learning rate, 2nd key as margin, 3rd key as epochs and value is error value")
            pprint(Error)
            keys = max(((r, mu, epochs) for r in Accuracy for mu in Accuracy[r] for epochs in Accuracy[r][mu]), key=lambda (r,mu,epochs):Accuracy[r][mu][epochs])            
            print("Best values are: (best learning rate, best margin, best epoch)")
            print(keys)
            test = 1
            Weight[keys[0]][keys[1]][keys[2]].insert(0,uniform(0,1))
            print("-------------------MARGIN PERCEPTRON RESULTS QN 3-------------------------------")
            print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
            Margine_Test_Result = Margin_perceptron(input_testset, keys[2], Weight[keys[0]][keys[1]][keys[2]], keys[0], keys[1])
            print("Test Result : \n %s" %(Margine_Test_Result))
            Margine_Training_Result = Margin_perceptron(inputset, keys[2], Weight[keys[0]][keys[1]][keys[2]], keys[0], keys[1])
            print("Training Result : \n %s" %(Margine_Training_Result))
            test = 0
            
        # QUESTION 3
        elif(decide_wts == 1 and ques == 4):
            #learning_rates = [1,0.1,0.01,0.001,0.0001,0.00001]
            mus = [1,0.1,0.001,4,5]
            epochss = [3,5]
            global flag
            flag = 0
            for times in range(2):
                Accuracy = {}
                Error = {}
                Weight = {}
                '''
                for i in range(125):
                    w = w + [uniform(0,1)]
                '''
                trainingset = parseTraininga2a(1, trainfile)
                inputset = create_inputvector(trainingset)
                for mu in mus:
                    Accuracy[mu] = {}
                    Error[mu] = {}
                    Weight[mu] = {}
                    for epochs in epochss:
                        for i in range(125):
                            w = w + [uniform(0,1)]
                        Result = Aggressive_perceptron(inputset, epochs, w, mu)
                        Accuracy[mu][epochs] = Result['accuracy']
                        Error[mu][epochs] = Result['Errors']
                        Weight[mu][epochs] = Result['Final weight vector']
                print("-------------------AGGRESSIVE PERCEPTRON TRAINING-------------------------------")
                if(flag == 1):
                    print("RESULTS OF QN 4 AFTER SHUFFELING ARE:")
                print("Accuracy for Aggressive perceptron in Qn 4 with 1st key as margin, 2nd key as epochs and value is accuracy value")
                pprint(Accuracy)
                print("Errors for Aggressive perceptron in Qn 4 with 1st key as margin, 2nd key as epochs and value is error value")
                pprint(Error)
                keys = max(((mu, epochs) for mu in Accuracy for epochs in Accuracy[mu]), key=lambda (mu,epochs): Accuracy[mu][epochs])
                print("Best values are: (best margin, best epoch)")
                print(keys)
                testset = parseTraininga2a(1, testfile)
                input_testset = create_inputvector(testset)
                #print Weight[keys[0]][keys[1]]
                #print len(Weight[keys[0]][keys[1]])
                test = 1
                Weight[keys[0]][keys[1]].insert(0,uniform(0,1))
                print("-------------------AGGRESSIVE PERCEPTRON RESULTS-------------------------------")
                print("Results are in form of dictionary with 1st value as no. of errors, 2nd is final weight vector, 3rd is accuracy")
                Aggressive_Test_Result = Aggressive_perceptron(input_testset, keys[1], Weight[keys[0]][keys[1]], keys[0])
                print("Test Result : \n %s" %(Aggressive_Test_Result))
                Aggressive_Training_Result = Aggressive_perceptron(inputset, keys[1], Weight[keys[0]][keys[1]], keys[0])
                print("Training Result : \n %s" %(Aggressive_Training_Result))
                test = 0
                flag = 1

main()
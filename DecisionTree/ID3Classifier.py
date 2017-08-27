#from pprint import pprint
import csv
import math
#import pydot

#graph = pydot.Dot(graph_type='graph')
TOTAL = 22
''' k could be used to read every second or third or .. line '''
def parseTraining(k, name):
    data = []
    with open(name, 'rb') as csvfile:
        counter = 0
        spamreader = csv.reader(csvfile)
        for line in spamreader:
            if(counter % k == 0):
                data.append(line)
            counter += 1
    return data

def calculateEntropy(p, e, total):
    entropy = 0
    for value in p.keys():
        if(not(p[value] == 0 or e[value] == 0)):
            entropy += ((p[value] + e[value]) / total) * ( - (p[value] / total) * math.log((p[value] / total), 2) -  (e[value] / total) * math.log((e[value] / total), 2))
    return entropy

''' entropy that we have bu dividing the dateset '''
def initialEntropy(data):
    p = 0.0
    e = 0.0
    for row in data:
        if(row[len(row) - 1] == 'p'):
            p += 1
        else:
            e += 1
    total = p + e
    if( p == 0.0 or e == 0.0):
        return 0.0
    return ( - (p / total) * math.log((p / total), 2) -  (e / total) * math.log((e / total), 2))

''' count the final decision a 'p' in dataset '''
def countP(data):
    counter = 0
    for line in data:
        if(line[len(line) - 1] == 'p'):
            counter += 1
    return counter

def makeDecisionTree(data, validColumns, limitdepth):
    if(TOTAL - len(validColumns) == limitdepth):
        node = Node()
        node.isLeaf = True
        if len(data) - countP(data) > countP(data):
            node.decision = 'e'
        else:
            node.decision = 'p'
        return node
    maxGain = -123123213.0
    maxEntropyColumn = 0
    maxEntropyValues = []
    entropy = initialEntropy(data)
    #print("entropy = %s " % (entropy))
    if entropy == 0.0:
        node = Node()
        node.isLeaf = True
        node.decision = data[0][len(data[0]) - 1]
        return node
    for column in validColumns:
        p = {}
        e = {}
        rowsLater = []
        for row in data:
            if row[column] == '?':
                rowsLater.append(row)
                continue
            if row[column] not in e:
                e[row[column]] = 0.0
            if row[column] not in p:
                p[row[column]] = 0.0
            if(row[len(row) - 1] == 'e'):
                e[row[column]] += 1
            else:
                p[row[column]] += 1
        ''' replacing the ? '''
        maxAttriCount = 0
        maxAttri = '?'
        #-----------------------------------------Method1 Start---------------------------------
        '''
        for k in p.keys():
            if(p[k] + e[k] > maxAttriCount):
                maxAttriCount = p[k] + e[k]
                maxAttri = k
        
        '''
        #-----------------------------------------Method1 End------------------------------------

        #-----------------------------------------Method2 Start----------------------------------
        '''
        ecount= 0.0
        pcount = 0.0
        for row in rowsLater:
            if(row[len(row) - 1] == 'e'):
                ecount += 1
            else:
                pcount += 1
        #print ecount
        #print pcount
        
        if(ecount>pcount): 
            maxAttri = max(e,key=e.get)
        else:
            maxAttri = max(p,key=p.get)
        '''
        #----------------------------------------Method2 End-------------------------------------
        #-----------------------------------------Common Start-----------------------------------
        '''
        for row in rowsLater:
            if(row[len(row) - 1] == 'e'):
                e[maxAttri] += 1
            else:
                p[maxAttri] += 1
        '''
        #-----------------------------------------Common End--------------------------------------

        ''' Calculating entropy of current column'''
        entropyCurrent = calculateEntropy(p, e, len(data))
        if(entropy - entropyCurrent > maxGain):
            maxEntropyColumn = column
            maxEntropyValues = p.keys()
            maxGain = entropy - entropyCurrent
    ''' Check for +- 1 '''
    node = Node()
    #print("maxEntropyColumn = %s " % (maxEntropyColumn))
    tempColumns = [number for number in validColumns if number != maxEntropyColumn]
    node.column = maxEntropyColumn
    node.values = maxEntropyValues
    #pprint(node.childrens)
    for value in maxEntropyValues:
        temp = [row for row in data if row[maxEntropyColumn] == value]
        #print("for children %s of column %s" % (value, maxEntropyColumn))
        childrenValue = makeDecisionTree(temp, tempColumns, limitdepth)
        #print("finished for children %s of column %s" % (value, maxEntropyColumn))
        node.childrens[value] = childrenValue
        #edge = pydot.Edge(str(node), str(childrenValue), label=value)
        #graph.add_edge(edge)
    return node

class Node:

    def __init__(self):
        self.childrens = {}
        self.column = 0
        self.values = []
        self.isLeaf = False
        self.decision = ''

    def __str__(self):
        if(self.isLeaf):
            return self.decision
        return str(self.column) + "_" + str(id(self))

def getDecision(root, data):
    if(root == None):
        return ''
    elif(root.isLeaf):
        return root.decision
    elif(data[root.column] not in root.childrens):
        #print data[root.column]
        return ''
    else:
        return getDecision(root.childrens[data[root.column]], data)

def findAccuracy(tree, testName):
    correct = 0.0
    length = 0.0
    with open(testName, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for line in spamreader:
            decision = getDecision(tree, line)
            if(decision == line[len(line) - 1]):
                correct += 1
            length += 1
        return correct / length

def maxDepth(root):
    if(root.isLeaf):
        return 0
    else :
        depths = []
        for v in root.childrens.values():
            depths.append(maxDepth(v))
        return max(depths) + 1

def settingABC():
    ''' Ques 1 '''
    
    validColumns = range(0, 22)
    data = parseTraining(1, 'datasets/SettingA/training.data')
    #print data
    root = makeDecisionTree(data, validColumns, 23)
    print("Maximum depth = %s" % (maxDepth(root)))
    #print("Accuracy = %s" % (findAccuracy(root, 'datasets/SettingA/test.data')))
    print("Accuracy = %s" % (findAccuracy(root, 'datasets/SettingA/training.data')))
    #graph.write_png('graph.png')
    
    #------------------------------------- Cross Validation Start------------------------------------------------

    
    depths = [1, 2, 3, 4, 5, 10, 15, 20]
    datas = [parseTraining(1, 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data') for i in range(0,6)]
    accu = {}
    average = {}
    stand_deviation = {}
    for depth in depths:
        accu[depth] = {}
    for depth in depths:
        sum_accu = 0.0
        for i in range(0, 6):
            data = []
            for k in range(0, 6):
                if k != i:
                    data = data + datas[k]
            root = makeDecisionTree(data, range(0, 22), depth)
            accu[depth][i] = findAccuracy(root, 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data')
            sum_accu = sum_accu + accu[depth][i]
        average[depth] = sum_accu/6.0 
    for depth in depths:
        sum_stand = 0.0
        for i in range(0, 6): 
            sum_stand = sum_stand + math.pow((accu[depth][i] - average[depth]),2)
        stand_deviation[depth] = math.sqrt(sum_stand/6.0) 
    #pprint(accu)
    print("Average Cross Validation Accuracy = %s" % (average))
    print("Standard Deviation = %s" % (stand_deviation))
    
    #---------------------------------------------Cross Validation End------------------------------------------
def main():
    settingABC()


main()

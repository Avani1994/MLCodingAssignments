import sys

###
# for changing the option being run change option argument in function fix_missing_in_data(c_data, option=2) to 1,2 or 3
#
#

from math import *
from random import shuffle
from random import sample
from random import choice
import svm_hand_tree as sv
from operator import itemgetter

attributes = {}
for i in range(0,500):
    attributes[i+1] = [0, 1]
    

'''attributes = { 1: ['b', 'c', 'x', 'f', 'k', 's'],
               2: ['f', 'g', 'y', 's'],
               3: ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
               4: ['t', 'f'],
               5: ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
               6: ['a', 'd', 'f', 'n'],
               7: ['c', 'w', 'd'],
               8: ['b', 'n'],
               9: ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
               10: ['e', 't'],
               11: ['b', 'c', 'u', 'e', 'z', 'r', '?'], #removed ?
               12: ['f', 'y', 'k', 's'],
               13: ['f', 'y', 'k', 's'],
               14: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
               15: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
               16: ['p', 'u'],
               17: ['n', 'o', 'w', 'y'],
               18: ['n', 'o', 't'],
               19: ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
               20: ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
               21: ['a', 'c', 'n', 's', 'v', 'y'],
               22: ['g', 'l', 'm', 'p', 'u', 'w', 'd']
}

label = ['e', 'p']
#label = [1, -1]

'''



def std_dev(l):
    m = (sum(l) + 0.0) / len(l)
    ssq = 0.0
    for c in l:
        ssq += (c - m)**2
    return sqrt(ssq/len(l))

def read_from_file(filename):
    f = open(filename, 'r')
    content_list = []
    for line in f.readlines():
        content_list.append(line.strip('\n').split(','))
    return content_list





def fix_missing_in_data(c_data, option=2):
    """Fixes the missing data according to the 3 options"""
    co = 0
    for row in c_data:
       lab = row[-1]
       missing = [i for i, x in enumerate(row) if x == '?']
       for ind in missing:
           atrr = attributes[ind+1]
           count = [0] * len(atrr)
           if option == 1:
                for r in c_data:
                    if r[ind] != '?':
                       count[atrr.index(r[ind])] += 1
                #print "Count: ", count       
                row[ind] = atrr[count.index(max(count))]
           if option == 2:
                for r in c_data:
                    if r[ind] != '?' and lab == r[-1]:
                        co += 1
                        count[atrr.index(r[ind])] += 1
                row[ind] = atrr[count.index(max(count))]
           if option == 3:
               return c_data
    #print "co", co                        
    return c_data


def count_attribute_numbers(number, c_list):
    """Counts the number of attr_val of each attribute in dataset"""
    total_count = [0, 0, 0]
    labels = attributes[number]
    #print labels
    counts = dict((l, [0, 0, 0]) for l in labels)
    #print counts
    for line in c_list:
        #print line[number-1]
        total_count[0] += 1
        counts[line[number-1]][0] += 1
        if line[-1] == -1:                    ###### negetive -1#####
            counts[line[number-1]][1] += 1
            total_count[1] += 1
        else:
            counts[line[number-1]][2] += 1
            total_count[2] += 1

    return counts, total_count

def calculate_log(p1, p2, total):
    if total == 0:
        return 0.0
    prob1 = p1/(total + 0.0)
    prob2 = p2/(total + 0.0)
    en = 0.0
    if prob1 != 0.0:
        en += -1 * prob1 * log(prob1, 2)
    if prob2 != 0.0:
        en += -1 * prob2 * log(prob2, 2)
    return en

def gen_subset_of_content(c_list, attr_val, value):
    subset = []

    for i in range(len(c_list)):
        if c_list[i][attr_val - 1] == value:
            subset.append(c_list[i])
    return subset

#print gen_subset_of_content(content_list, 1, 'c')

def calculate_entropy(attribute_count, total_count):
    entropy = 0.0
    for key in attribute_count:
        entropy += ((attribute_count[key][0] + 0.0) / total_count[0]) * calculate_log(attribute_count[key][1], attribute_count[key][2], attribute_count[key][0])

    cal_en = calculate_log(total_count[1], total_count[2], total_count[0])

    return cal_en - entropy


def find_max_gain(attr_to_consider, c_list, k = 11):
    IGs = []
    ##################################################ONLY PICK K random elements#########
    if len(attr_to_consider) >= 11:
        attr_to_consider = sample(attr_to_consider, k)
    ######################################################################################
    for attr in attr_to_consider:
        attribute_count, total_count = count_attribute_numbers(attr, c_list)
        IGs.append(calculate_entropy(attribute_count, total_count))
    #print max(IGs), IGs
    return max(IGs), attr_to_consider[IGs.index(max(IGs))]

#print find_max_gain(list(range(1,23)), content_list)

class Node():
    def __init__(self, attribute_split, depth=0):
        self.value = attribute_split
        self.children = []
        self.attr_val = []
        self.depth = depth

    def __str__(self):
        #return str(self.value) + str(self.children) + str(self.label)
        return str(self.value)


def calculate_max_classified(s):
    c_p = 0
    c_e = 0
    for line in s:
        if line[-1] == -1:   ###negetive -1 'e'####
            c_e += 1
        else:
            c_p += 1
    if c_p >= c_e:
        return 1      ### return positive####
    else:
        return -1     #### return negetive####


### GLOBAL VARIABLE TO KEEP TRACK OF CURRENT DEPTH ####
depths = 0


def ID3(S, attribute, depth=0, limit=0, limitOn=False):
    #print 'attribute: ', attribute, S==None
    global depths
    #print 'limit: ', limit, depth, depths
       
    if not S:
        #print "not S"
        return
    
    #print "Depth: ", depth
    if depths <= depth:
        depths = depth
    
        
    if limitOn == True:
       if limit == (depth):
           return Node(calculate_max_classified(S), depth = depth)
    
    #if depths < depth:
    #    depths = depth
    #check if all examples have same label
    if not attribute:
        return
    
    #print len(S)
    lab = S[0][-1]
    all_same = True
    for line in S[1:]:
       if line[-1] != lab:
           all_same = False
           #break
    if all_same:
        #print "All are same"
        return Node(lab, depth=depth)

    # Oh! all were not same
    

    #find the best
    num, attr = find_max_gain (attribute, S)
    #print "best attr: ", attr
    root = Node(attr)
    values = attributes[attr]
    attribute_copy = attribute[:]
    attribute_copy.remove(attr)
    for value in values:
        #print "Now considering value number: ", value
        root.attr_val.append(value)
        # generate subset
        s = gen_subset_of_content(S, attr, value)
        #print "len of S and s: ", len(S), len(s)
        if s == None:
            #print "none"
            #root.children.append(Node('p')) #modify this
            root.children.append(Node(calculate_max_classified(S)))
        else:
            #print "attribute12: ", attribute
            ro= ID3(s, attribute_copy, depth+1, limit, limitOn)
            root.children.append(ro)
    return root


def print_tree(n):
    if n == None:
        return
    
    # We're at the leaf
    if n.value == 1 or n.value == -1:     ### negetive
        #print n
        return 
    
    if not n.children:
        return
    i = 0
    for child in n.children:
        print n, str(n.attr_val[i]),  child
        i += 1
        print_tree(child)

#print_tree(root)
#print "depth of tree: ", depths

def check_example(example, root, gtruth):
    if root == None:
         return 
    #print root.value, example, root.label, root.children
    if root.value == 1 or root.value == -1:             #### negetive
        if root.value == gtruth:
            return True
        else:
            return False
    
    #print root.label.index(example[root.value - 1]), root.label
    if root.children[root.attr_val.index(example[root.value - 1])] is None:
        return False
    l = root.children[root.attr_val.index(example[root.value - 1])].value
    #print 'l', l
    
    
    if l == None:
        return False
    else:
        m = root.children[root.attr_val.index(example[root.value - 1])]
        if isinstance(m, Node):
            #print 'm:', m
            return check_example(example, m, gtruth)
        else:
            if m == gtruth:
                return True
            else:
                return False    

def get_output(example, root):
    if root == None:
         return 
    #print root.value, example, root.label, root.children
    if root.value == 1 or root.value == -1:             #### negetive
        return root.value
    
    #print root.label.index(example[root.value - 1]), root.label
    if root.children[root.attr_val.index(example[root.value - 1])] is None:
        return choice([1, -1]) ### unable to predict return random return false 
    l = root.children[root.attr_val.index(example[root.value - 1])].value
    #print 'l', l
    
    
    if l == None:
        return choice([1, -1]) # return False
    else:
        m = root.children[root.attr_val.index(example[root.value - 1])]
        if isinstance(m, Node):
            #print 'm:', m
            return get_output(example, m)
        else:
            return m  



def test_accuracy(content_test, root):
    total = 0
    correct = 0
    for example in content_test:
        #print "example: ", total
        if check_example(example, root, example[-1]):
            correct += 1
        total +=  1
    return (correct + 0.0 ) * 100.0 / total



def check_train_and_test_accuracy(root, train_data, test_data, dep):
    #print depths
    
    #root = ID3(train_data, list(range(1,23)), 0, dep, True)
    #print print_tree(root)
    #print "depths: ", depths
    return test_accuracy(train_data, root), test_accuracy(test_data, root) #, depths


def read_file(filename):
    """ reads the file and return data"""
    f = open(filename)
    data = []
    for line in f.readlines():
        line = line.strip("\n").strip(" ").split(" ")
        int_line = []
        for val in line:
            #print val
            int_line.append(float(val))
        data.append(int_line)
    return data
    
def read_label_file(filename):
    """ reads the file and return data"""
    f = open(filename)
    data = []
    for line in f.readlines():
        line = line.strip("\n")
        data.append(int(line))
    return data    
        

#folder = "./handwriting/"
#folder = "./madelon/madelon_"
#train_d = read_file("train.data")
#train_l = read_label_file("train.labels")

#test_d = read_file("test.data")
#test_l = read_label_file("test.labels")


'''
train_data = []
test_data = []
i = 0
for i in range(len(train_d)):
     l = train_d[i]
     l.append(train_l[i])
     train_data.append(l)
#print train_data
    
for i in range(len(test_d)):
     l = test_d[i]
     l.append(test_l[i])
     test_data.append(l)
#print test_data'''


#root = ID3(train_data, list(range(1,257)))

#print check_train_and_test_accuracy(root, train_data, test_data, 100)
'''
print "OUTPUTS : "

for example in train_data:
    print get_output(example, root) 

print "END OUTPUTS "
'''
### pick out only m random samples

def return_only_m_samples(data, m):
     return sample(data, m)
     
#print len(train_data)
#print len(return_only_m_samples(train_data, 15))     
    
def train_n_trees(train_data, n, m):
    #sset = sample(train_data, m)
    trees = []
    for i in range(n):
        sset = [choice(train_data) for _space in range(m)]
        root = ID3(sset, list(range(1,501)))
        trees.append(root)
    return trees




def prepare_output(data, roots):
    outputs = []
    labels = []
    for example in data:
        out = []
        for root in roots:
           out.append(get_output(example, root))
        labels.append(example[-1])   
        outputs.append(out)
    return {'data': outputs, 'label': labels}

def parse_for_svm(trfile,lfile):
    
    for i,row in enumerate(trfile):
        row.extend([lfile[i]])
        row.insert(0, 1)
    #print trfile
    return trfile




#print svm.train_and_test_svm(tr_data, tr_l, te_data, te_l,  epoch=1, C=1, g_i=0.01)


#***************************************SPLITTING****************************************************************
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
        labels = labels + [int(line)]

    for i,row in enumerate(data):
        row.extend([labels[i]])
        row.insert(0, 1)
    
    return data

def infogain(data_set):
    #print data_set[1]
    sort_data = []
    for i in range(1,501):
        labels = []
        lists = []
        #print data_set[0]
        for row in range(1,len(data_set)):
            #print type(data_set[row][501])
            labels.append(data_set[row][501])
            lists.append(data_set[row][i])
        #print lists
        #print labels
        yx = zip(lists, labels)    
        yx.sort()
        sort_data = sort_data + [yx] 

    #print len(sort_data)
    return sort_data

def calculate_split(sort_data):
    gini_array = []
    c=0
    count1 = 0.0
    countminus1 = 0.0
    max_infogain = [0.0] * 500
    #print max_infogain
    split = [-1] * 500
    for line in sort_data:
        if(line[501] == '1'):
            count1 = count1 +1.0
        else:
            countminus1 = countminus1 + 1.0

    gini_label = 1 - (count1/2000.0 * count1/2000.0) - (countminus1/2000.0 * countminus1/2000.0)           

    '''for line in sort_data:
        print line
        for i,element in enumerate(line):
            print element[i],"....................."
            print element[i][1], element[i+1][1], "Avani"
            break
            if(element[i][1]!=element[i+1][1]):
                gini_split = 1 - ((i+1)/2000.0 * (i+1)/2000.0) - ((2000-i-1)/2000.0 * (2000-i-1)/2000.0)
                info_gain = gini_label - gini_split
                if(info_gain > max_infogain):
                    max_infogain = info_gain
                    split = i+1
    print max_infogain, "hi"
    print split,"o"
    '''
    #c=0
    #info_gain=0
    #print len(sort_data[450])
    for i in range(500):
        #print sort_data[i][1]
        for j in range(2000-2):
            if sort_data[i][j][1]!=sort_data[i][j+1][1]:
            #for i,element in enumerate(line):
                #print element[i],"....................."
                #print sort_data[i][j], sort_data[i][j+1], "Avani"
                #break
                gini_split = 1 - ((j)/2000.0 * (j)/2000.0) - ((2000-j)/2000.0 * (2000-j)/2000.0)
                info_gain = gini_label - gini_split

                if(info_gain > max_infogain[i]):
                    max_infogain[i] = info_gain
                    split[i] = sort_data[i][j][0]
                    c=c+1
                    #print j
        #print info_gain,max_infogain, "hi"
    #print max_infogain
    #print split
    #print len(split)
    return split

def make_trainfile_testfile_forDtree(train_input, test_input,split):
    for line in train_input:
        #print(len(line))  
        for i, element in enumerate(line[1:501]):
            if(element > split[i]):
                line[i+1] = 1.0
            else:
                line[i+1] = 0.0
    for line in test_input:    
        for i, element in enumerate(line[1:501]):
            if(element > split[i]):
                line[i+1] = 1.0
            else:
                line[i+1] = 0.0

    return {'train':train_input, 'test':test_input}

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
        trees = [10,30,100]
        #print train_input
        #shuffle(train_input)
        sortt_data = infogain(train_input)
        splitt = calculate_split(sortt_data)
        files = make_trainfile_testfile_forDtree(train_input, test_input, splitt) 
        final_train = files['train']
        final_test = files['test']
        train_data = []
        test_data = []
        i = 0
        for i in range(len(final_train)):
            l = final_train[i][1 :]
            train_data.append(l)
        #print len(train_data), "train"

        for i in range(len(final_test)):
            l = final_test[i][1 :]
            
            test_data.append(l)
        #print len(test_data), "test"

        
        '''
        for root in roots:
            print test_accuracy(test_data, root)
        '''
        accura = []
        print("qn 3.2.2")
        print("Training Results")
        for n in trees:
            roots = train_n_trees(train_data, n, 1500)
            result = prepare_output(train_data, roots)
            tr_data = result['data']
            tr_l = result['label']
            
            #print tr_data
            #print tr_l

            train_input = parse_for_svm(tr_data, tr_l)
            #test_input = parse_for_svm(te_data, te_l)
            #print(len(train_input)), "avnu"
            final_wt = sv.trainsvm(train_input,1,n)

            Resulttrain = sv.accuracy(final_wt, train_input,n)
            #Resulttest = sv.accuracy(final_wt,test_input)
            accura.append([Resulttrain['accur'],n,final_wt,roots])

            print Resulttrain,"for",n,"trees"
            
        #print accura
        accura.sort(key=itemgetter(0), reverse = True)
        #print accura
        final_roots = accura[0][3]
        result2 = prepare_output(test_data, final_roots)
        te_data = result2['data']
        te_l = result2['label']
        test_input = parse_for_svm(te_data, te_l)
        #print(len(test_input))
        #print len(accura[0][2])
        Resulttest = sv.accuracy(accura[0][2],test_input,accura[0][1])
        print("Testing Results")
        print Resulttest, "TEST RESULTS for", accura[0][1], "trees" 

main()

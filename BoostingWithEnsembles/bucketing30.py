from random import randint
from random import shuffle
from math import *
from random import shuffle
from random import sample
from random import choice
from random import seed
#import svm

seed()

f=open("madelon_train.data",'r')
f=f.readlines()
train=[]
for line in f:
  train.append(line.strip().split(' '))

print len(train),len(train[0])
#print train[1]

f=open("madelon_train.labels",'r')
f=f.readlines()
trainlabel=[]
for line in f:
  trainlabel.append(line.strip().split('\n'))
  
#print trainlabel[0]

f=open("madelon_test.data",'r')
f=f.readlines()
test=[]
for line in f:
  test.append(line.strip().split(' '))

#print test[0]


f=open("madelon_test.labels",'r')
f=f.readlines()
testlabel=[]
for line in f:
  testlabel.append(line.strip().split('\n'))


def buck(train):
  maximum=0
  minimum=0
  rang=0
  a1=[]
  aa=[]
  for j in range(0,500):
    a=[]
    for i in train:
      a.append(int(i[j]))
    #print a  
    minimum=min(a)
    #print minimum
    maximum=max(a)
    #print maximum
    rang=maximum-minimum
    #print rang
    rang=rang/10
    #print rang
    for i in a:
      #print i
      if i<minimum+rang:
        a1.append(0)
      elif i<minimum+2*rang:
        a1.append(1)
      elif i<minimum+3*rang:
        a1.append(2)
      elif i<minimum+4*rang:
        a1.append(3)
      elif i<minimum+5*rang:
        a1.append(4)
      elif i<minimum+6*rang:
        a1.append(5)
      elif i<minimum+7*rang:
        a1.append(6)
      elif i<minimum+8*rang:
        a1.append(7)
      elif i<minimum+9*rang:
        a1.append(8)
      else:
        a1.append(9)
    #print a[i],a1[i]  
    #break
  
    #print a[i]
    #print a1
    aa.append(a1)    
    a1=[]
  #print len(aa)  
  #print len(aa[1])
  #print aa[0]

  aa11=[list(x) for x in zip(*aa)]
  #print len(aa11)  
  #print len(aa11[1])
  #print aa11[0]
  return aa11


train1=buck(train)
test1=buck(test)

  


###
# for changing the option being run change option argument in function fix_missing_in_data(c_data, option=2) to 1,2 or 3
#
#


attributes = {}
for i in range(0,500):
    attributes[i+1] = [0,1,2,3,4,5,6,7,8,9]
    



def std_dev(l):
    m = (sum(l) + 0.0) / len(l)
    ssq = 0.0
    for c in l:
        ssq += (c - m)**2
    return sqrt(ssq/len(l))

def read_from_file(filename):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        data.append(line.strip('\n').split(','))
    return data



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

'''
def find_max_gain(attr_to_consider, c_list, k = 8):
    IGs = []
    #considering log_2 d=8 attributes
    if len(attr_to_consider) >= 8:
        attr_to_consider = sample(attr_to_consider, k)
    
    for attr in attr_to_consider:
        attribute_count, total_count = count_attribute_numbers(attr, c_list)
        IGs.append(calculate_entropy(attribute_count, total_count))
    #print max(IGs), IGs
    return max(IGs), attr_to_consider[IGs.index(max(IGs))]
'''
def find_max_gain(attr8, c_list, k = 11):
    IGs = []
    #considering log_2 d=11 attributes
    if len(attr8) >= 11:
        attr8 = sample(attr8, k)
    
    for attr in attr8:
        attribute_count, total_count = count_attribute_numbers(attr, c_list)
        IGs.append(calculate_entropy(attribute_count, total_count))
    #print max(IGs), IGs
    return max(IGs), attr8[IGs.index(max(IGs))]


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
        
folder=""
#folder = ".data/handwriting/"
#folder = "./madelon/madelon_"
train_d = read_file(folder + "train.data")
train_l = read_label_file(folder + "train.labels")

test_d = read_file(folder + "test.data")
test_l = read_label_file(folder + "test.labels")



train_data = []
test_data = []
i = 0
for i in range(len(train1)):
     l = train1[i]
     l.append(float(trainlabel[i][0]))
     train_data.append(l)
#print train_data
    
for i in range(len(test1)):
     l = test1[i]
     l.append(float(testlabel[i][0]))
     test_data.append(l)
#print test_data


root = ID3(train_data, list(range(1,501)))
'''
print check_train_and_test_accuracy(root, train_data, test_data, 100)

print "OUTPUTS : "

for example in train_data:
    #print get_output(example, root)
    print "" 

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
        sset=[choice(train_data) for a in range(m)]
        root = ID3(sset, list(range(1,501)))
        trees.append(root)
    return trees


roots = train_n_trees(train_data, 30, 900)
'''
for root in roots:
    print test_accuracy(test_data, root)
'''

def prepare_output(data, roots):
    outputs = []
    labels = []
    for example in data:
        out = []
        for root in roots:
           out.append(get_output(example, root))
        labels.append(example[-1])   
        outputs.append(out)
    return outputs, labels



#print prepare_output(train_data, roots)
train, trainlabel = prepare_output(train_data, roots)
test, testlabel = prepare_output(test_data, roots)

#print svm.train_and_test_svm(tr_data, tr_l, te_data, te_l,  epoch=1, C=1, g_i=0.01)

T=30
w=[0]*31
z=0
#print z
gamma=0.1
C=0.03125
x1=0
gamma0=gamma

def mulvec(x,y):
  q=[0]*31
  for i in range(len(y)):
    q[i]=x[i]*y[i]
  return q

def mul(x,y):
  q=[0]*31
  #print x,y
  for i in range(len(y)):
    q[i]=x*float(y[i])
  return q

def addvec(x,y):
  q=[0]*31
  for i in range(len(y)):
    q[i]=x[i]+y[i]
  return q

dd=1.0
mis=0
for t in range(0,T):
  z=0
  #print t
  #shuffle data
  train_shuf = []
  trainlabel_shuf = []
  index_shuf = range(len(train))
  shuffle(index_shuf)
  for i in index_shuf:
    train_shuf.append(train[i])
    trainlabel_shuf.append(trainlabel[i])
  #i=randint(0,len(train))
  #print i
  for i in range(len(train_shuf)):
    x=[]
    #x.append(x1)
    x=train_shuf[i]
    #print "length",len(x)
    x=[1]+x
    #x.insert(0,1)
    #for j in range(0)
    #print x
    #print "length",len(x)
    #print x 
    y=trainlabel_shuf[i]
    #y=str(trainlabel[i]).strip('\n').split(' ')
    #y=int(y[0])
    #y=float(y[0])
    #print y
    #print len(y)
    #z=float(x[0])*float(w[0])
    z=0
    for j in range(len(x)):
      #print j,x[j],z[j]
    
      z+=float(x[j])*float(w[j])
    #print z
    #for j in range(0,256):
    #  #print j,x[j],z[j]
    #  
    #  z1[j]=float(y)*float(z[j])
 
    if y*z<=1:
      qq=gamma0*C
      x1=mul((1-gamma0),w)
      x2=mul(y,x)
      x3=mul(qq,x2)
      w=addvec(x1,x3)
      mis=mis+1
    else:
      w=mul((1-gamma0),w) 
    gamma0=gamma/float(1+(gamma*float(dd))/C)
    #dd=1
    #print dd
    dd=dd+1

#print w  

#accuracy

#print len(train)
mistake=0
tp=0.0
fp=0.0
fn=0.0
p=0.0
r=0.0
f1=0.0
tot=0
for i in range(0,2000):
  sum1=0
  x=train[i]
  x=[1]+x
  #print 
  y=trainlabel[i]
  #print float(y[0])
  for j in range(0,31):
    sum1+=float(x[j])*w[j]
  
  #sum1+=b
  if sum1<0:
    sum1=-1
  else:
    sum1=1
  if sum1==1 and float(y)>0:
    tp=tp+1
  if sum1==1 and float(y)<0:
    fp=fp+1
  if sum1==-1 and float(y)>0:
    fn=fn+1
  if float(y)*sum1<=0:
    mistake=mistake+1
  tot=tot+1
if(tp != 0.0 or fp != 0.0):
  p = tp / (tp + fp)
if(tp != 0.0 or fp != 0.0):
  r = tp / (tp + fn)
if(p != 0.0 or r != 0.0):
  f1 = 2.0 * p * r/(p + r)

print "TRAINING RESULTS"
print "training accuracy",((2000-mistake)/2000.0)*100  
#print mistake, "train mistake"
print "precision",p
print "recall",r
print "fscore",f1

#  if float(y[0])*sum1<=0:
#    mistake=mistake+1
#print mistake  
#for j in range(0,256):
#  #print j,x[j],z[j]
    
#  z+=float(x[j])*float(w[j])
#print len(test)
mistake=0
tp=0.0
fp=0.0
fn=0.0
p=0.0
r=0.0
f1=0.0
tot=0
for i in range(0,600):
  sum1=0
  x=test[i]
  x=[1]+x
  #print 
  y=testlabel[i]
  #print float(y[0])
  for j in range(0,31):
    sum1+=float(x[j])*w[j]
  #sum1+=b
  if sum1<0:
    sum1=-1
  else:
    sum1=1
  if sum1==1 and float(y)>0:
    tp=tp+1
  if sum1==1 and float(y)<0:
    fp=fp+1
  if sum1==-1 and float(y)>0:
    fn=fn+1
  if float(y)*sum1<=0:
    mistake=mistake+1
  tot=tot+1
if(tp != 0.0 or fp != 0.0):
  p = tp / (tp + fp)
if(tp != 0.0 or fp != 0.0):
  r = tp / (tp + fn)
if(p != 0.0 or r != 0.0):
  f1 = 2.0 * p * r/(p + r)

print "TESTING RESULTS"
print "testing accuracy",((600-mistake)/600.0)*100  
#print mistake, "train mistake"
print "precision",p
print "recall",r
print "fscore",f1


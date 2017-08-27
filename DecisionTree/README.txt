CADE Machine on which Program was tested :LAB1-35

I have included screenshots for my outputs of crossvalidation on Setting A, Setting B, Method 1, Method 2 and Method 3 for Setting C. Second Last Dictionary in ouput is Average Accuracy of depths and Last Dictionary is Standard Deviation. If you want to run actual experiments you should follow following guidelines.

So main file is ID3Classifier.py
This can be easily run using shell script ./treeclassifier.sh
1st run of ./treeclassifier.sh gives reults for normal testing and training with training.data of Setting A. It also gives Crossvalidation for CV splits of A. For further settings follow given guidelines.
*************************************************************************************************************************
For Normal Testing
*************************************************************************************************************************
Training files and Test files are taken directly from the code. So they need to be changed according to the question in the file ID3Classifier.py. 
For line no. 198 and 203 in function SettingABC()
which are:
data = parseTraining(1, 'datasets/SettingA/training.data')
print("Accuracy = %s" % (findAccuracy(root, 'datasets/SettingA/training.data')))

if we are training on training.data of Setting B and testing on test.data of Setting B
path in line 198 must be changed to 'datasets/SettingB/training.data' 
and path in line 203 must be changed to 'datasets/SettingB/test.data'

Default Path is
For line 198 'datasets/SettingA/training.data'
For line 203 'datasets/SettingA/training.data'

Program outputs Accuracy and Depth. Error can be calculated as 1 - Accuracy.
 
**************************************************************************************************************************
For CrossValidation
**************************************************************************************************************************
For Cross Validation
Open file ID3Classifier.py
Uncomment Lines under
#--------------------Cross validation Start----------------- till #------------------Cross validation End------------------

For CrossValidation too the desired setting must be changed in the file ID3Classifier.py.

For line no. 210 and 224 in function SettingABC()
which are:
datas = [parseTraining(1, 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data') for i in range(0,6)]
accu[depth][i] = findAccuracy(root, 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data')

if we are training and testing on CVSplits of Setting B 
path in line 210 must be changed to 'datasets/SettingB/CVSplits/training_0' + str(i) + '.data' 
and path in line 224 must be changed to 'datasets/SettingB/CVSplits/training_0' + str(i) + '.data'

Default Path is
For line 210 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data'
For line 224 'datasets/SettingA/CVSplits/training_0' + str(i) + '.data'


**********************************************************************************************************************
For Method 1, Method 2 and Method 3 in Setting C
**********************************************************************************************************************
For Method 1:
Open file ID3Classifier.py
Uncomment the Lines under 
#--------------------Method1 Start-------------------------- till #-----------------------Method1 End------------------
and under
#--------------------Common Start--------------------------- till #-----------------------Common End-------------------
then save and run ./treeclassifier.sh

For Method 2:
Uncomment the Lines under 
#--------------------Method2 Start-------------------------- till #-----------------------Method2 End-------------------
and under
#--------------------Common Start--------------------------- till #-----------------------Common End--------------------
then save and run ./treeclassifier.sh

For Method 3:
Simply, the training data lines containing missing values are being ingnored. 
Which is being done in line no. 73 to line no. 75. No need to uncomment any lines.

************************************************************************************************************************
Limiting Depth
************************************************************************************************************************
To limit the depth, a variable named limitdepth is being passed in makeDecisionTree(data, validColumns, limitdepth).
Suppose we want to limit depth to 4.

then in ID3Classifier.py
line no. 200 which is 
root = makeDecisionTree(data, validColumns, 23)
pass 4 instead of 23.

If you want tree to go till maximum depth pass any number greater than 22 because tree can have maximum length of 22 as 
there are 22 attributes.

Default Depth is 23 to allow tree to grow till its maximum depth.

***********************************************************************************************************************
Extra Libraries used
***********************************************************************************************************************
NONE

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++END+++++++++++++++++++++++++++++++++++++++++++++++++++++










clear
chmod u+x runme.sh
python svmhand.py train.data train.labels test.data test.labels train
python svmmadelon.py madelon_train.data madelon_train.labels madelon_test.data madelon_test.labels 
python hand_ensemble.py
python madelon_ensemble.py madelon_train.data madelon_train.labels madelon_test.data madelon_test.labels
python bucketing10.py
python bucketing30.py
python bucketing100.py

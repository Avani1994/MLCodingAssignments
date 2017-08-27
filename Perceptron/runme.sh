#!/bin/bash
python perceptron.py table2 0 1 a5a.test > outputQn1.txt
python perceptron.py a5a.train 1 2 a5a.test > outputQn2.txt
python perceptron.py a5a.train 1 3 a5a.test > outputQn3.txt
python perceptron.py a5a.train 1 4 a5a.test > outputQn4.txt
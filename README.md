# mnist

In this repo, the MNIST dataset (in data folder) is used to build three classifiers to classify handwritten digits.

- loadmnist.py load the MNIST dataset and split the training set to three subsets X1, X2 and X3. Then the three subsets and the testing sets are stored locally to "mnist.pkl.gz". 
- classifiers.py use three classifiers to classify the digits, then the majority voting algorithm is used to combine these results.
- this code is run in python 3. 

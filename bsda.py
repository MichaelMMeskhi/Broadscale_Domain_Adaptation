#!/usr/bin/env python3

"""
Multi-Layer Neural Network and Support Vector Machine
Using active learning with UncertaintySampling
Author: Michael M.Meskhi
Project: Broadscale Domain Adaptation with Active Learning
Date: 2018-06-01
"""

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


def results(accuracy, model_name):
    print(model_name)
    print("==========================")
    print("Mean Accuracy: " + str(round(np.mean(accuracy)*100, 2)) + "\n" + "Standard Deviation: " + str(round(np.std(accuracy), 2)))
    print("==========================\n")


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out, accuracy = [], [], []

    for _ in range(quota):
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        accuracy = np.append(accuracy, model.score(tst_ds))

    return E_in, E_out, accuracy


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
    trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds

def main():
    ''' @sys.argv[1] : Absolute path to dataset. 
        @sys.argv[2] : Validation set size.
        @sys.argv[3] : Number of pre-labled points in target dataset.
        @sys.argv[4] : Query budget. 
    '''

    # Path to your libsvm_sparse type classification dataset.
    # If dataset not in libsvm_sparse type use libsvm to convert.
    dataset_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), sys.argv[1])
    test_size = float(sys.argv[2])   # The percentage of samples in the dataset that will be randomly selected and assigned to the test set.
    n_labeled = int(sys.argv[3])   # Number of samples that are initially labeled.

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = int(sys.argv[4])   # Number of samples to query.

    # Model is the base learner, e.g. LogisticRegression, SVM ... etc.
    models = {'MLP': SklearnProbaAdapter(MLPClassifier(hidden_layer_sizes=(10,10,10,10,10,10), solver='lbfgs', alpha=2, random_state=1, activation='relu')), 
            'SVM': SVM(gamma=0.001)}
    
    for model_name, model in models.items():
        qs = UncertaintySampling(trn_ds, method='lc', model=model)
        E_in_1, E_out_1, accuracy = run(trn_ds, tst_ds, lbr, model, qs, quota)

        # Plot the learning curve of UncertaintySampling.
        # The x-axis is the number of queries, and the y-axis is the corresponding error rate.
        query_num = np.arange(1, quota + 1)
        plt.plot(query_num, E_in_1, 'b', label='qs Ein')
        plt.plot(query_num, E_out_1, 'g', label='qs Eout')
        plt.xlabel('Number of Queries')
        plt.ylabel('Error')
        plt.title('Experiment Result')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        plt.show()

        # Plot the accuracy over requested queries. 
        # The x-axis is the number of queries, and the y-axis is the corresponding accuracy rate.
        filename = 'results/' + model_name + '.png'
        plt.plot(query_num, accuracy, 'y', label="accuracy")
        plt.xlabel('Number of Queries')
        plt.ylabel('Accuracy')
        plt.title(model_name + ' + Active Learning')
        plt.legend(loc='upper center', bbox_to_anchor=(0.8, -0.5), fancybox=True, shadow=True, ncol=5)
        plt.savefig(filename)
        plt.show()

        results(accuracy, model_name)

if __name__ == '__main__':
    main()
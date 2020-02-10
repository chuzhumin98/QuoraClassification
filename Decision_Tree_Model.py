from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl

from dataloader import save_object
import numpy as np
import os

def F1_measure(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    precision = float(np.sum(labels[predictions == 1])) / float(np.sum(predictions))
    recall = float(np.sum(predictions[labels == 1])) / float(np.sum(labels))
    F1 = 2*precision*recall / (precision + recall)
    print('precision = {}, recall = {}, F1 = {}'.format(precision, recall, F1))
    print(np.sum(predictions[labels == 1]), np.sum(predictions[labels == 0]), np.sum(labels[predictions == 0]))


if __name__ == '__main__':
    with open('data/train_words_attributes_100.pkl', 'rb') as file:
        train_words_attributes = pkl.load(file)

    with open('data/valid_words_attributes_100.pkl', 'rb') as file:
        valid_words_attributes = pkl.load(file)

    with open('data/train_label.pkl', 'rb') as file:
        train_labels = pkl.load(file)

    with open('data/valid_label.pkl', 'rb') as file:
        valid_labels = pkl.load(file)

    with open('data/test_words_attributes_100.pkl', 'rb') as file:
        test_words_attributes = pkl.load(file)

    path_clf =  'cache/decision_tree_base.pkl' # 'cache/random_forest_base.pkl'

    if path_clf == 'cache/decision_tree_base.pkl':
        if not os.path.exists(path_clf):
            tree_model = tree.DecisionTreeClassifier()
            clf = tree_model.fit(train_words_attributes, train_labels)

            save_object(clf, path_clf)
        else:
            with open(path_clf, 'rb') as file:
                clf = pkl.load(file)
    else:
        if not os.path.exists(path_clf):
            tree_model = RandomForestClassifier(n_estimators=20)
            clf = tree_model.fit(train_words_attributes, train_labels)

            save_object(clf, path_clf)
        else:
            with open(path_clf, 'rb') as file:
                clf = pkl.load(file)

    print(len(valid_labels))



    predict_valid = clf.predict(valid_words_attributes)
    print(predict_valid)

    print(np.sum(np.abs(np.array(valid_labels) - np.array(predict_valid))))

    print(sum(predict_valid))

    F1_measure(predict_valid, valid_labels)

    predict_test = clf.predict(test_words_attributes)
    print(predict_test.shape)

    # save_object(predict_test, './result/rf_base.pkl')
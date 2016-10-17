import numpy as np
from sklearn.ensemble import EnsembleSelectionClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing

import pandas as pd

# load dataset
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

original_data = pd.read_csv(
    "/path/to/adult.data/",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
encoded_data, _ = number_encode_features(original_data)
X = encoded_data[encoded_data.columns.difference(["Target"])]
y = encoded_data["Target"]

# split dataset
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.95, random_state=0)
X_train, X_val, y_train, y_val = cross_validation.train_test_split(
    X_train, y_train, test_size=0.2, random_state=0)

scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype(np.float64)),
                       columns=X_train.columns)
X_val = scaler.transform(X_val.astype(np.float64))
X_test = scaler.transform(X_test.astype(np.float64))

print X_train.shape

n_features = X.shape[1]
C = 1.0
# Create different classifiers. The logistic regression cannot do
# multiclass out of the box.
classifiers = [('L1 logistic', LogisticRegression(C=C, penalty='l1')),
               ('L2 logistic (OvR)', LogisticRegression(C=C, penalty='l2')),
               ('L2 logistic (Multinomial)', LogisticRegression(
                C=C, solver='lbfgs', multi_class='multinomial')),
               ('Linear SVC 10', SVC(kernel='linear', C=10, probability=True,
                                 random_state=0)),
               ('Linear SVC 1', SVC(kernel='linear', C=C, probability=True,
                                 random_state=0)),
               ('Linear SVC 0.1', SVC(kernel='linear', C=0.1,
                                     probability=True, random_state=0)),
               ('Linear SVC 10^-2', SVC(kernel='linear', C=0.01,
                                      probability=True, random_state=0)),
               ('Linear SVC 10^-3', SVC(kernel='linear', C=0.001,
                                       probability=True, random_state=0)),
               ('Linear SVC 10^-4', SVC(kernel='linear', C=0.0001,
                                       probability=True, random_state=0)),
               ('Boosted DT 16', AdaBoostClassifier(
                                 DecisionTreeClassifier(max_depth=7),
                                 n_estimators=16)),
               ('Boosted DT 32', AdaBoostClassifier(
                                 DecisionTreeClassifier(max_depth=7),
                                 n_estimators=32)),
               ('Boosted DT 64', AdaBoostClassifier(
                                 DecisionTreeClassifier(max_depth=7),
                                 n_estimators=64)),
               ('Boosted DT 128', AdaBoostClassifier(
                                 DecisionTreeClassifier(max_depth=7),
                                 n_estimators=128))]

n_classifiers = len(classifiers)
print "Validation Set:"
for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_val)
    y_pred = y_pred_proba.argmax(axis=1)
    classif_rate = accuracy_score(y_val, y_pred)
    print("classif_rate for %s : %f " % (name, classif_rate))

print "Test Set:"
for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    #y_pred_proba = classifier.predict_proba(X_test)
    #y_pred = y_pred_proba.argmax(axis=1)
    y_pred = classifier.predict(X_test)
    classif_rate = accuracy_score(y_test, y_pred)
    print("classif_rate for %s : %f " % (name, classif_rate))


esc = EnsembleSelectionClassifier(estimators=classifiers, n_bags=20, n_best=1,
                                  bag_fraction=0.5, verbose=True)

esc.fit(X_val, y_val)
y_pred = esc.predict(X_test)
classif_rate = accuracy_score(y_test, y_pred)
print("classif_rate for EnsembleSelectionClassifier : %f " % classif_rate)
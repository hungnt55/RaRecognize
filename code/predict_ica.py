from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random as rnd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import sor_wc_wk_joint as sor
import cv_split
import time
import pickle
from sklearn.decomposition import FastICA

scaler = StandardScaler()

pfile = 'test_data/test_' + sys.argv[1] + '.npz'

params = np.load(pfile)

train_classes = params['train_classes']
test_classes = params['test_classes']
train_index = params['train_index']
test_index = params['test_index']

risk_class_files = ['Drought.csv', 'Earthquakes.csv', 'Explosions.csv', 'Floods.csv', 'Forest_and_Brush_Fire.csv', 'Hazardous_and_Toxic_Substance.csv', 'Landslides.csv', 'Lighting.csv', 'Snowstorms.csv', 'Tornado.csv', 'Tropical Storms.csv', 'Volcanoes.csv', 'Water_Pollution.csv']

risk_class_dict = {}
for i in range(len(risk_class_files)):
  risk_class_dict[i+1] = risk_class_files[i]

def remove_label(docs):
  for i in range(len(docs)):
    docs[i] = docs[i].replace('"1, ','').replace('"0, ','').replace("'0, ",'').replace("'0, ",'')
  return docs

risk_classes = {}
for risk_file in risk_class_files:
  risk_classes[risk_file] = pd.read_csv('../data/NYTimes_data/'+risk_file, header = None)[0].tolist()

non_risk_file = 'non_risk_docs.csv'

non_risk_class = pd.read_csv('../data/NYTimes_data/'+non_risk_file, header = None)[0].tolist()

X = []
Y = []

class_id = 1

for risk_file in risk_class_files:
  X += risk_classes[risk_file]
  Y += [class_id] * len(risk_classes[risk_file])
  class_id += 1

X += non_risk_class
Y += [0] * len(non_risk_class) 

X = remove_label(X)

tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

features = tfidf.fit_transform(X).toarray()
labels = Y

def run_test(features, labels, train_classes, test_classes, train_index, test_index):
    cpu_time = 0
    features = np.array(features)
    labels = np.array(labels)
    xtrain = features[np.isin(labels,train_classes),:]
    ytrain = labels[np.isin(labels,train_classes)]

    xtest = features[np.isin(labels,test_classes),:]
    ytest = labels[np.isin(labels,test_classes)]

    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

    scaler.fit(X_train)
    ica = pickle.load(open('test_data/'+sys.argv[1] + '_ica.sav', 'rb'))

    X_train = ica.transform(scaler.transform(X_train))
    X_test = ica.transform(scaler.transform(X_test))
    xtest = ica.transform(scaler.transform(xtest))

    y_test_l = y_test.tolist()

    model = sor.HierarchicalClassifierModel(input_size = X_train[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, l1 = 0, l2 = 0, train_classes = train_classes)

    model_s = pickle.load(open('test_results/trained_model_' + sys.argv[1] + '_ica_joint' + '.m', 'rb'))
  
    model.copy(model_s)
    
    model.evt_fit_threshold(X_train, y_train)

    y_pred = model.predict(X_test, 0)
    np.savetxt('test_results/' +sys.argv[1] + '_ica_joint_seen.pred', y_pred)

    for classk in range(len(test_classes)):
        print('test class', test_classes[classk])
        xtest_ri = xtest[ytest == test_classes[classk]]
        y_pred_ri = model.predict(xtest_ri, 0)
        np.savetxt('test_results/' + sys.argv[1] + '_ica_joint_unseen_' + str(classk)+ '.pred', y_pred_ri)

run_test(features, labels, train_classes, test_classes, train_index, test_index)
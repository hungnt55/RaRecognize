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
import time
from sklearn.decomposition import FastICA
import pickle

scaler = StandardScaler()

pca = PCA(0.99)

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
  print(risk_file)
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

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,1), stop_words='english', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

features = tfidf.fit_transform(X).toarray()
labels = Y

def run_setup(features, labels, train_classes, test_classes, test_fold):
    features = np.array(features)
    labels = np.array(labels)
    xtrain = features[np.isin(labels,train_classes),:]
    ytrain = labels[np.isin(labels,train_classes)]

    xtest = features[np.isin(labels,test_classes),:]
    ytest = labels[np.isin(labels,test_classes)]

    train_ids = np.arange(len(ytrain))
    np.random.shuffle(train_ids)

    train_index = train_ids[:int(len(ytrain)*0.8)]
    test_index = train_ids[int(len(ytrain)*0.8+1):]

    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

    model = sor.HierarchicalClassifierModel(input_size = X_train[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, model_name = 'wSOR', l1 = 0.1, l2 = 0)

    model.fit_wk(X_train, y_train)

    model.save('test_data/trained_model_'+str(test_fold) +'_wk.m')

    np.savez('test_data/test_' + str(test_fold) + '.npz', train_classes = train_classes, test_classes = test_classes, train_index = train_index, test_index = test_index)

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
    features = tfidf.fit_transform(X).toarray()
    features = np.array(features)
    xtrain = features[np.isin(labels,train_classes),:]
    ytrain = labels[np.isin(labels,train_classes)]

    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    pca.fit(X_train_std)

    model = sor.HierarchicalClassifierModel(input_size = pca.transform(X_train_std)[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, model_name = 'wSOR', l1 = 0.1, l2 = 0)

    model.fit_wk(pca.transform(X_train_std), y_train)

    model.save('test_data/trained_model_'+str(test_fold) + '_pca_wk.m')

    transformer = FastICA(n_components=pca.n_components_, random_state=0, max_iter=500)
    xtrain_transformed = transformer.fit_transform(X_train_std)

    model = sor.HierarchicalClassifierModel(input_size = transformer.transform(X_train_std)[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, model_name = 'wSOR', l1 = 0.1, l2 = 0)

    model.fit_wk(transformer.transform(X_train_std), y_train)

    model.save('test_data/trained_model_'+str(test_fold) +'_ica_wk.m')

    pickle.dump(transformer, open('test_data/'+str(test_fold) + '_ica.sav', 'wb'), protocol=4)  


class_ids = np.arange(14)
for test_fold in range(5):
  np.random.shuffle(class_ids[1:])
  print(class_ids)

  test_classes = class_ids[10:14]
  train_classes = np.array([i for i in range(14) if i not in test_classes])

  print('Train classes:', train_classes)
  print('Test classes:', test_classes)

  run_setup(features, labels, train_classes, test_classes, test_fold)
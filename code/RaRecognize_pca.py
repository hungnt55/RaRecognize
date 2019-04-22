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
import sor_wc_wk_joint_noredundant as sor
import time
import cv_split
import pickle

scaler = StandardScaler()

pca = PCA(0.99)

pfile = 'test_data/test_' + sys.argv[1] + '.npz'

params = np.load(pfile)

print(params.files)

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
    features = np.array(features)
    labels = np.array(labels)
    xtrain = features[np.isin(labels,train_classes),:]
    ytrain = labels[np.isin(labels,train_classes)]

    xtest = features[np.isin(labels,test_classes),:]
    ytest = labels[np.isin(labels,test_classes)]

    RR = 0
    R2Ri = 0
    RNR = 0
    NRNR = 0
    NRR = 0
    RiR = np.zeros(len(test_classes))
    RiNR = np.zeros(len(test_classes))

    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]

    scaler.fit(X_train)
    pca.fit(scaler.transform(X_train))
    X_train = pca.transform(scaler.transform(X_train))
    X_test = pca.transform(scaler.transform(X_test))
    xtest = pca.transform(scaler.transform(xtest))

    y_test_l = y_test.tolist()

    model = sor.HierarchicalClassifierModel(input_size = X_train[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, l1 = 0, l2 = 0, train_classes = train_classes)

    parameters = {'l1':np.logspace(-2,2,5), 'l2':np.append(np.logspace(-3,1,5),0)}
    splitter = cv_split.UnseenTestSplit()
    cmodel = GridSearchCV(model, parameters, cv = splitter, verbose=5, n_jobs = 10, error_score='raise')
    cmodel.fit(X_train, y_train.ravel())

    print('best params: l1=', cmodel.best_params_['l1'], 'l2=', cmodel.best_params_['l2'])

    model = sor.HierarchicalClassifierModel(input_size = X_train[0].size, num_classes = len(risk_class_files), learning_rate = 1e-3, num_epochs = 1000, batch_size = 100, l1 = cmodel.best_params_['l1'], l2 = cmodel.best_params_['l2'], train_classes = train_classes)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test, 0)
    y_pred_score = model.predict_score(X_test, 0)
    np.savetxt('test_results/' + sys.argv[1] + '_pca_joint_seen.out', y_pred_score)

    for j in range(len(y_test)):
            if y_test_l[j] >= 1:
                if y_pred[j] == 1:
                    RR += 1
                    y_pred_class = model.predict(X_test[j,:].reshape(1,-1), int(y_test[j]))
                    if y_pred_class == 1:
                      R2Ri += 1
                else:
                    RNR += 1
            else:
                if y_pred[j] == -1:
                    NRNR += 1
                else:
                    NRR += 1

    for classk in range(len(test_classes)):
            xtest_ri = xtest[ytest == test_classes[classk]] 
            y_pred_ri = model.predict(xtest_ri, 0)
            y_pred_ri_score = model.predict_score(xtest_ri, 0)
            np.savetxt('test_results/' + sys.argv[1] + '_pca_joint_unseen_' + str(classk)+ '.out', y_pred_ri_score)

            for j in range(len(y_pred_ri)):
              if y_pred_ri[j] == 1:
                RiR[classk] += 1
              else:
                RiNR[classk] += 1

    print(RR, RNR, NRR, NRNR, RiR, RiNR, R2Ri)
    pickle.dump(model, open('test_results/trained_model_' + sys.argv[1] + '_pca_joint' + '.m', 'wb'))

print('Train classes:', train_classes)
print('Test classes:', test_classes)

run_test(features, labels, train_classes, test_classes, train_index, test_index)

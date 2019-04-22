from sklearn.base import BaseEstimator, ClassifierMixin
import data_helpers
import numpy as np
from imblearn.over_sampling import SMOTE
import pickle

class HierarchicalClassifierModel(BaseEstimator, ClassifierMixin):
  def __init__(self, input_size = 1000, num_classes = 17, num_epochs = 1000, batch_size = 100, learning_rate = 0.001, l1 = 1e-3, l2 = 1e-5, model_name = 'wSOR', train_classes = 0):

    self.input_size = input_size
    self.num_classes = num_classes
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.l1 = l1
    self.l2 = l2
    self.model_name = model_name
    self.weight = np.random.randn(input_size, num_classes+1)
    self.bias = np.random.randn(num_classes+1)
    self.train_classes = train_classes

  def loss_wk(self, X, Y, k):
    return np.sum(np.maximum(1-Y*(np.dot(X, self.weight[:,k]) + self.bias[k]),0)) + self.l1 * np.sum(self.weight[:,k] ** 2)/2

  def loss_wc(self, X, Y, l2, train_classes):
    w2 = self.weight**2
    sum_w = np.sum(w2[:,train_classes], 1) 
    X_norm = np.linalg.norm(X[0], axis=0) ** 2
    return self.loss_wk(X, Y, 0) + l2 * np.sum((sum_w/2 + w2[:,0]/4) * w2[:,0] * X_norm) 
 
  def loss_overall(self, X, Y, train_classes):
    w2 = self.weight**2
    sum_w = np.sum(w2[:,train_classes], 1) 
    X_norm = np.linalg.norm(X[0], axis=0) ** 4
    return np.sum([self.loss_wk(X[k], Y[k], k) for k in train_classes]) + self.loss_wk(X[0], Y[0], 0) + self.l2 * np.sum( ( (sum_w/2 + w2[:,0]/4) * w2[:,0] + np.sum(self.weight[:,train_classes]**4,1)/4) * X_norm) 
 
  def fit_wk(self, X_train, Y_train):
    sm = SMOTE(random_state=2, k_neighbors = 2)

    # Train the model
    X_train_r = X_train[Y_train != 0]
    Y_train_r = Y_train[Y_train != 0]
    for class_k in np.unique(Y_train):
      if class_k == 0:
        continue
      print("Training class " + str(class_k))

      #print(Y_train_r)
      Y_traink = np.copy(Y_train_r)
      Y_traink[Y_train_r != class_k] = -1
      Y_traink[Y_train_r == class_k] = 1

      X_train_rk, Y_train_rk = sm.fit_sample(X_train_r, Y_traink.ravel())
      best_loss = self.loss_wk(X_train_rk, Y_train_rk, class_k)
      loss = 0
      count_loss_increase = 0
      learning_rate = self.learning_rate

      for epoch in range(1,self.num_epochs+1):
        outputs = np.dot(X_train_rk, self.weight[:,class_k]) + self.bias[class_k]
        outputs = Y_train_rk * [Y_train_rk*outputs < 1]

        grad_w = np.sum(-X_train_rk.T * outputs,1) + self.l1 * self.weight[:, class_k]
        grad_b = np.sum(-outputs)

        self.weight[:, class_k] -= learning_rate * grad_w
        self.bias[class_k] -= learning_rate * grad_b

        loss = self.loss_wk(X_train_rk, Y_train_rk, class_k)
        print(loss)
        if loss >= best_loss:
          learning_rate /= 2
          if count_loss_increase > 50:
            break
          count_loss_increase += 1
        else:
          best_loss = loss


  def fit_all(self, X_train, Y_train):

    sm = SMOTE(random_state=2, k_neighbors = 3)

    if self.model_name == 'wSOR':
      # Train the model
      X_train_r = X_train[Y_train != 0]
      Y_train_r = Y_train[Y_train != 0]
      for class_k in np.unique(Y_train):
        if class_k == 0:
          continue
        print("Training class " + str(class_k))

        #print(Y_train_r)
        Y_traink = np.copy(Y_train_r)
        Y_traink[Y_train_r != class_k] = -1
        Y_traink[Y_train_r == class_k] = 1

        print(self.score(X_train_r, Y_traink, class_k))

        X_train_rk, Y_train_rk = sm.fit_sample(X_train_r, Y_traink.ravel())
        best_loss = self.loss_wk(X_train_rk, Y_train_rk, class_k)
        loss = 0
        count_loss_increase = 0
        learning_rate = self.learning_rate

        for epoch in range(1,self.num_epochs+1):
          outputs = np.dot(X_train_rk, self.weight[:,class_k]) + self.bias[class_k]
          outputs = Y_train_rk * [Y_train_rk*outputs < 1]

          grad_w = np.sum(-X_train_rk.T * outputs,1) + self.l1 * self.weight[:, class_k]
          grad_b = np.sum(-outputs)

          self.weight[:, class_k] -= learning_rate * grad_w
          self.bias[class_k] -= learning_rate * grad_b

          loss = self.loss_wk(X_train_rk, Y_train_rk, class_k)
          print(loss)
          if loss >= best_loss:
            learning_rate /= 2
            if count_loss_increase > 10:
              break
            count_loss_increase += 1
          else:
            best_loss = loss
  
        print(self.score(X_train_r, Y_traink, class_k))

      self.retrain_wc(X_train, Y_train, self.l2)

    else:
      Y_trainc = np.copy(Y_train)
      Y_trainc[Y_train != 0] = 1
      Y_trainc[Y_train == 0] = -1

      print("Training class 0")

      print(self.score(X_train, Y_trainc, 0))

      X_trainc, Y_traincc = sm.fit_sample(X_train, Y_trainc.ravel())

      sum_w = np.sum(self.weight[:,1:]**2,1)
 
      x2 = np.dot(X_trainc.T, X_trainc) ** 2
      best_loss = self.loss_wc(X_trainc, Y_traincc, 0)
      loss = 0
      count_loss_increase = 0
      learning_rate = self.learning_rate

      for epoch in range(1,self.num_epochs+1):
        outputs = np.dot(X_trainc, self.weight[:,0]) + self.bias[0]
        outputs = Y_traincc * [Y_traincc*outputs < 1]
      
        grad_w = np.sum(-X_trainc.T * outputs,1) + self.l1 * self.weight[:, 0]
        grad_b = np.sum(-outputs)

        self.weight[:, 0] -= learning_rate * grad_w
        self.bias[0] -= learning_rate * grad_b

        loss = self.loss_wc(X_trainc, Y_traincc, 0)
        print(loss)
        if loss >= best_loss:
          learning_rate /= 2
          if count_loss_increase > 10:
            break
          count_loss_increase += 1
        else:
          best_loss = loss

  def fit(self, X_train, Y_train):

    self.train_classes = np.array([i for i in np.unique(Y_train) if i != 0])

    X_k = X_train[Y_train != 0]
    Y_tmp = Y_train[Y_train != 0]

    X_c = np.copy(X_train)
    Y_c = np.copy(Y_train)
    Y_c[Y_c != 0] = 1
    Y_c[Y_c == 0] = -1

    X = {}
    Y = {}
    X[0] = np.copy(X_c)
    Y[0] = np.copy(Y_c)
    Y_k = np.copy(Y_tmp)
    sm = SMOTE(random_state=2, k_neighbors = 2)
    for k in self.train_classes:
      Y_k[Y_tmp == k] = 1
      Y_k[Y_tmp != k] = -1
      #X_traink, Y_traink = sm.fit_sample(X_k, Y_k.ravel())
      X_traink, Y_traink = X_k, Y_k
      X[k] = np.copy(X_traink)
      Y[k] = np.copy(Y_traink)

    X_norm = 0

    if self.l2 != 0:
      X_norm = np.linalg.norm(X[0], axis=0) ** 4

    best_loss = self.loss_overall(X, Y, self.train_classes)
    loss = 0
    learning_rate = self.learning_rate
    test_rate = True
    count_loss_increase = 0

    for epoch in range(1,self.num_epochs+1):

      weight_tmp = np.copy(self.weight)
      bias_tmp = np.copy(self.bias)
      w2 = self.weight**2

      # gradient updates for wk
      for k in self.train_classes:
        outputs = np.dot(X[k], weight_tmp[:,k]) + bias_tmp[k]
        outputs = Y[k] * [Y[k]*outputs < 1]

        grad_w = np.sum(-X[k].T * outputs,1) + (self.l1 + self.l2 * (w2[:,0] + w2[:,k]) * X_norm) * weight_tmp[:,k] 
        grad_b = np.sum(-outputs)

        self.weight[:,k] -= learning_rate * grad_w
        self.bias[k] -= learning_rate * grad_b

      # gradient updates for wc

      outputs = np.dot(X[0], weight_tmp[:,0]) + bias_tmp[0]
      outputs = Y[0] * [Y[0]*outputs < 1]

      grad_w = np.sum(-X[0].T * outputs,1) + (self.l1 + self.l2 * (np.sum(w2[:,self.train_classes],1) + w2[:,0])*X_norm) * weight_tmp[:,0]

      grad_b = np.sum(-outputs)

      self.weight[:,0] -= learning_rate * grad_w
      self.bias[0] -= learning_rate * grad_b

      loss = self.loss_overall(X, Y, self.train_classes)

      if loss >= best_loss :
        if count_loss_increase > 50:
          break
        self.weight = np.copy(weight_tmp)
        self.bias = np.copy(bias_tmp)
        count_loss_increase += 1
        learning_rate /= 2
      else:
        if loss > best_loss * (1 - 1e-5):
          count_loss_increase += 1
        else:
          count_loss_increase = 0
        best_loss = loss

    print(self.score(X_train, Y_train))
        
  def save(self, name):
    # Save the model checkpoint
    pickle.dump(self.weight, open(name+'_w', 'wb'))
    pickle.dump(self.bias, open(name+'_b', 'wb'))

  def load(self, name):
    self.weight = pickle.load(open(name+'_w', 'rb'))
    self.bias = pickle.load(open(name+'_b', 'rb'))

  def score(self, X, Y):
    Ys = np.copy(Y)

    outputs = self.predict(X, 0)
    outputs_r = outputs[Ys >= 1]
    outputs_nr = outputs[Ys == 0]
    precision = 0
    if ((outputs_r == 1).sum().item() + (outputs_nr == 1).sum().item()) != 0:
      precision = (outputs_r == 1).sum().item()*1.0/((outputs_r == 1).sum().item() + (outputs_nr == 1).sum().item())
    recall = (outputs_r == 1).sum().item()*1.0/len(outputs_r)
    print('precision', precision)
    print('recall', recall)
    if (precision != 0) or (recall != 0):
      return 2*precision*recall/(precision+recall)
    else:
      return 0

  def predict(self, X, k):
    outputs = np.dot(X, self.weight[:,k]) + self.bias[k]
    return np.sign(outputs)
  
  def predict_score(self, X, k):
    outputs = np.dot(X, self.weight[:,k]) + self.bias[k]
    return outputs
  
class HierarchicalClassifierModel_test:
  def __init__(self, iters, l1, l2):
    np.set_printoptions(threshold=np.nan)
    x = np.random.random((200,1000))
    y = np.zeros(200, dtype=int)
    for i in range(0,10):
      y[i*20:i*20+20] = np.ones(20, dtype=int)*i
      x[i*20:i*20+20,i*20:i*20+20] += 2
      x[i*20:i*20+20,10*20:10*20+20] += 2
      print(x[i*20:i*20+20,i*20:i*20+20])
    x[0:20,10*20:10*20+20] -= 2
    print(y)
    model = HierarchicalClassifierModel(num_epochs = iters, num_classes = 9, learning_rate = 0.001, model_name = 'wSOR', l1 = l1, l2 = l2)
    model.fit(x,y)
    print(model.score(x,y))
    print(model.weight.T)

#model = HierarchicalClassifierModel_test()

#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[49]:


# Read Data
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
X = np.array(iris.drop(['species'], axis=1))
y = np.array(iris['species'])
# todo: balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5,random_state=1)
iris.head()


# In[50]:


iris.describe()


# In[34]:


iris['species'].value_counts()


# In[35]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


# In[36]:


class MyGaussianNB:
    def __init__(self):
        self.categories = None
        self.priors  = list()
        self.means = None
        self.vars = None
    
    def _calc_means(self, X, y):
        means = list()
        for  cate in self.categories:
            means.append(np.mean(X[y == cate, :], axis=0))
        self.means = np.vstack(means)
    
    def _calc_vars(self, X, y):
        vars = list()
        for cate in self.categories:
            vars.append(np.var(X[y == cate, :], axis=0))
        self.vars = np.vstack(vars)
    
    def _calc_likelihood(self, x):
        hoods = list()
        for i in range(len(self.categories)):
            probs = 1 / (np.sqrt(2 * np.pi) * np.sqrt(self.vars[i, :])) * np.exp(- (x - self.means[i, :]) ** 2 / (2 * self.vars[i, :]))
            hood = np.prod(probs)
            hoods.append(hood)
        return hoods
    
    def _calc_prior(self, y):
        # P(Y_i)
        for cate in self.categories:
            self.priors.append(np.sum(y == cate) / len(y))
    
    def _calc_posterior(self, x):
        posteriors = list()
        hoods = self._calc_likelihood(x)
        for i in range(len(self.categories)):
            posteriors.append(hoods[i] * self.priors[i])  
        return posteriors
                            
    def fit(self, X, y):
        self.categories = np.unique(y).tolist()
        self._calc_means(X, y)
        self._calc_vars(X, y)
        self._calc_prior(y)
                               
    def predict(self, X):
        probs = list()
        for i in range(X.shape[0]):
            this_probs = self._calc_posterior(X[i, :])
            this_probs = np.array(this_probs) / np.sum(this_probs)
            probs.append(this_probs)
        probs = np.vstack(probs)
        preds = np.argmax(probs, axis=1)
        preds = [self.categories[i] for i in preds]
        return preds
    
    def predict_prob(self, X):
        probs = list()
        for i in range(X.shape[0]):
            this_probs = self._calc_posterior(X[i, :])
            this_probs = np.array(this_probs) / np.sum(this_probs)
            probs.append(this_probs)
        probs = np.vstack(probs)
        return pd.DataFrame(probs, columns=self.categories)
            


# In[37]:


mdl = MyGaussianNB()
mdl.fit(X_train, y_train)


# In[38]:


train_preds = mdl.predict(X_train)


# In[39]:


confusion_matrix(y_train, train_preds)


# In[40]:


test_preds = mdl.predict(X_test)


# In[41]:


confusion_matrix(y_test, test_preds)


# In[42]:


print(mdl.means)


# In[43]:


print(mdl.vars)


# In[44]:


test_preds = mdl.predict(X_test)
print(test_preds)


# In[ ]:





# In[ ]:





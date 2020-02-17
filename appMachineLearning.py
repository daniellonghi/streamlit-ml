######################################### IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, SGDClassifier, RidgeClassifier, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
######################################### IMPORTS

st.sidebar.header("Machine Learning App")
st.sidebar.text("by @DLFP - longhi88@hotmail.com")

######################################## CLASS
class appMLSupervisioned:
  originalDataset = ""
  dataItens = ""
  dataItensFiltered = ""
  targetColumns = 0
  columnsToPredict = ""
  whatRadioButton = ""
  sizeTest = ""
  columnsSelected = ""
  whichButton = ""
  numberOfColumns = 0
  dropColumns = ""
  radioRegression = ""
  radioClassifier = ""
  X_new = ""

  def loadFileMachineLearning(self):
    uploadFile = st.sidebar.file_uploader("Please, choose a CSV file", type="csv")
    if uploadFile is not None:
      self.originalDataset = pd.read_csv(uploadFile)
      st.header("Original Dataset")
      st.write(self.originalDataset.head(10))
      self.SuperUnsuper()

  def SuperUnsuper(self):
    #Verificar se eh Super ou Unsuper
    radioButtonSuperUnsuper = st.sidebar.radio(
      "What kind of ML would you like to use?",
      ("Supervisioned", "Unsupervisioned")
    )
    if radioButtonSuperUnsuper == "Supervisioned":
      st.write("a")
      self.dropAnyColumn()
    else:
      st.write("b")
      #self.getBestColumnsToTrainTest()

  def dropAnyColumn(self):
    if self.originalDataset is not None:
      self.dropColumns = st.sidebar.multiselect(
        "Drop Selected Column",
        self.originalDataset.columns
      )
      self.dataItens = self.originalDataset.drop(self.dropColumns, axis = 1)
      self.radioButtonBestColumn()

  def radioButtonBestColumn(self):
    radioButton = st.sidebar.radio(
      "Select or use Best Column(s)?",
      ("All Column(s)", "KBest Algorithm")
    )
    self.whatRadioButton = radioButton
    if radioButton == "All Column(s)":
      self.getTargetColumnsToPredict()
    else:
      self.getBestColumnsToTrainTest()

  def getBestColumnsToTrainTest(self):
    if self.whatRadioButton == "KBest Algorithm":
      self.numberOfColumns = len(self.dataItens.columns)
      self.columnsSelected = st.sidebar.slider(
        "Please, Select the number of Columns",
        1, (self.numberOfColumns - 1)
      )
    self.getTargetColumnsToPredict()
      
  
  def getTargetColumnsToPredict(self):
    st.header("Train/Test Dataset")
    self.targetColumns = st.sidebar.multiselect(
      "Target(s)",
      self.dataItens.columns
    )


    ##CALL FOR KBEST FUNCTION
    if len(self.targetColumns):
      self.dataItens = self.dataItens.drop(self.targetColumns, axis = 1)
      
      if self.whatRadioButton == "KBest Algorithm":
        self.X_new = SelectKBest(chi2, k = int(self.columnsSelected)).fit_transform(self.dataItens, self.originalDataset[self.targetColumns])
        st.write(self.X_new)
      else:
        st.write(self.dataItens)

      self.sizeTest()
    else:
      st.markdown("Please, Select a **Target** column of the dataset.")
    ##CALL FOR KBEST FUNCTION
      
  def sizeTest(self):
    if len(self.targetColumns):
      self.sizeTest = st.sidebar.text_input(
        "Select a size (0.2 is recommended)",
        0.2
      )
      self.chooseAlgorithm()

  def chooseAlgorithm(self):
      self.whichButton = st.sidebar.radio(
        "Regression or Classifier?",
        ("Regression", "Classifier")
      )
      if self.whichButton == "Regression":
        self.regressionAlgo()
        self.buttonTrainTest()
      else:
        self.classifierAlgo()
        self.buttonTrainTest()

  def classifierAlgo(self):
    if self.whichButton is not None:
      self.radioClassifier = st.sidebar.radio(
          "Select the Algorithm Classifier",
          (
            "DecisionTreeClassifier",
            "KNeighborsClassifier",
            "LogisticRegressionClassifier", 
            "SGDClassifier"
          )
        )

  def regressionAlgo(self):
    if self.whichButton is not None:
      self.radioRegression = st.sidebar.radio(
          "Select the Algorithm Regressor",
          (
            "LinearRegression",
            "SGDRegression",
            "LassoRegression",
            "RidgeRegression"
          )
        )
  def buttonTrainTest(self):
    if st.sidebar.button('Train/Test Model'):
      st.header("Trainning and Testing Model - Wait Please")
      X_train, X_test, y_train, y_test = self.createXyTrainTest()
      
      if self.whichButton == "Regression":
        st.write("Regression")
        if self.radioRegression == "LinearRegression":
          model, y_pred = self.linearRegressionModel(X_train, X_test, y_train, y_test)
          self.metricsRegression(model, y_pred, X_train, X_test, y_train, y_test)
        elif self.radioRegression == "SGDRegression":
          model, y_pred = self.SGDRegressionModel(X_train, X_test, y_train, y_test)
          self.metricsRegression(model, y_pred, X_train, X_test, y_train, y_test)
        elif self.radioRegression == "LassoRegression":
          model, y_pred = self.LassoClassifierModel(X_train, X_test, y_train, y_test)
          self.metricsRegression(model, y_pred, X_train, X_test, y_train, y_test)
        else:
          model, y_pred = self.RidgeClassifierModel(X_train, X_test, y_train, y_test)
          self.metricsRegression(model, y_pred, X_train, X_test, y_train, y_test)
      else:
        st.write("Classification")
        if self.radioRegression == "DecisionTreeClassifier":
          model, y_pred = self.DecisionTreeClassifier(X_train, X_test, y_train, y_test)
          self.metricsClassification(model, y_pred, X_train, X_test, y_train, y_test)
        elif self.radioRegression == "KNeighborsClassifier":
          model, y_pred = self.KNNModel(X_train, X_test, y_train, y_test)
          self.metricsClassification(model, y_pred, X_train, X_test, y_train, y_test)
        elif self.radioRegression == "LogisticRegressionClassifier":
          model, y_pred = self.LogisticClassifierModel(X_train, X_test, y_train, y_test)
          self.metricsClassification(model, y_pred, X_train, X_test, y_train, y_test)
        else:
          model, y_pred = self.SGDClassifierModel(X_train, X_test, y_train, y_test)
          self.metricsClassification(model, y_pred, X_train, X_test, y_train, y_test)

    else:
      st.write()

  def metricsClassification(self, model, y_pred, X_train, X_test, y_train, y_test):
    st.write("Para o modelo atual as métricas foram de:")
    st.text(classification_report(y_test, y_pred))

  def DecisionTreeClassifier(self, X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def KNNModel(self, X_train, X_test, y_train, y_test):
    #DEFAULT KNN =3
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def LogisticClassifierModel(self, X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def SGDClassifierModel(self, X_train, X_test, y_train, y_test):
    model = SGDClassifier(max_iter = 1000, tol = 1e-3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def createXyTrainTest(self):
    X = self.originalDataset.drop(self.dropColumns, axis = 1)
    X = X.drop(self.targetColumns, axis = 1)
    y = self.originalDataset[self.targetColumns]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return train_test_split(X, y, test_size = float(self.sizeTest))

  def linearRegressionModel(self, X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def SGDRegressionModel(self, X_train, X_test, y_train, y_test):
    model = SGDClassifier(max_iter = 5000, alpha = 0.1)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

  def RidgeClassifierModel(self, X_train, X_test, y_train, y_test):
    model = RidgeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
  
  def LassoClassifierModel(self, X_train, X_test, y_train, y_test):
    model = Lasso(alpha = 0.1)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
  
  def metricsRegression(self, model, y_pred, X_train, X_test, y_train, y_test):
    st.write("Para o modelo atual as métricas foram de:")
    st.write("R2: ", (float("{0:.2f}".format(r2_score(y_test, y_pred)))) * 100)
    st.write("MSE: ", (float("{0:.2f}".format(mean_squared_error(y_test, y_pred)))) * 100)
    st.write("MAE: ", (float("{0:.2f}".format(mean_absolute_error(y_test, y_pred)))) * 100)
    
    if (len(X_test.columns) == 1):
      originalTrain = np.linspace(min(X_train.values.ravel()),max(X_train.values.ravel()), len(X_train))[:, np.newaxis]
      originalTest = np.linspace(min(X_test.values.ravel()),max(X_test.values.ravel()), len(X_test))[:, np.newaxis]
      
      st.header("Graphics: ")
      #plt.scatter(X_train, y_train, color = 'b')
      plt.scatter(X_test, y_test, color = 'g')
      #plt.plot(originalTrain, y_train, color = 'b')
      plt.plot(originalTest, y_test, color = 'g')
      plt.plot(originalTest, model.predict(originalTest), color = 'k')
      plt.xlabel(self.targetColumns[0].capitalize())
      plt.ylabel(self.dataItens.columns[0].capitalize())
      st.pyplot()
######################################## CLASS


####################################### CALLABLE
appMLSupervisioned = appMLSupervisioned()
appMLSupervisioned.loadFileMachineLearning()
####################################### CALLABLE
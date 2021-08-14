import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import HTML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
from Logging.Logger import Logger

class Classification:
    def __init__(self):
        pass

    def fit(self, x_train, x_test, y_train, y_test):
        self.L = ["Logistic_Regression", "Random_Forest", "XGBoost"]
        self.Model = []
        self.Score = []
        data = []
        try:

            ## Logistic Regression

            self.log_reg = LogisticRegression()
            self.param_grid_Logistic = {

                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'tol': [0.01, 0.001, 0.0001],
                'max_iter': [100, 1000, 10000]

            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.log_reg, param_grid=self.param_grid_Logistic, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(x_train, y_train.values.ravel())

            # extracting the best parameters
            self.penalty = self.grid.best_params_['penalty']
            self.tol = self.grid.best_params_['tol']
            self.max_iter = self.grid.best_params_['max_iter']

            # creating a new model with the best parameters
            self.log_reg = LogisticRegression(penalty=self.penalty, tol=self.tol, max_iter=self.max_iter)
            # training the mew model
            self.log_reg.fit(x_train, y_train.values.ravel())
            self.prediction_log_reg = self.log_reg.predict(x_test)


            self.Model.append(self.log_reg)

            if len(y_test[
                       "output"].unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.log_reg_score = accuracy_score(y_test, self.prediction_log_reg)
                self.Score.append(self.log_reg_score)
            else:
                self.log_reg_score = roc_auc_score(y_test, self.prediction_log_reg , multi_class='ovo', average='weighted')
                self.Score.append(self.log_reg_score)

            Logger().log("Classification.py", "INFO", " Logistic Regression is performed on the Dataset ")
            with open('result/LogisticRegression_Model.pickle', 'wb') as f:
                pickle.dump(self.log_reg, f)
                Logger().log("Classification.py", "INFO",
                             "Saved LogisticRegression Model as LogisticRegression_Model.pickle")

            data.append(({
                'Accuracy': metrics.accuracy_score(y_test, self.prediction_log_reg),
                'Precision': metrics.precision_score(y_test, self.prediction_log_reg),
                'Recall or Sensitivity': metrics.recall_score(y_test, self.prediction_log_reg),
                'F1_Score': metrics.f1_score(y_test, self.prediction_log_reg),
                'roc_auc_score': roc_auc_score(y_test, self.prediction_log_reg, multi_class='ovo', average='weighted'),
                "Model_Download": '<a href="//result/LogisticRegression_Model.pickle" download="LogisticRegression_Model.pickle">LogisticRegression_Model  </a>'
            }))

            ## Random Forest
            self.clf = RandomForestClassifier()

            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(x_train, y_train.values.ravel())

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(x_train, y_train.values.ravel())
            self.prediction_random_forest = self.clf.predict(x_test)  # prediction using the Random Forest Algorithm


            self.Model.append(self.clf)

            if len(y_test[
                       "output"].unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(y_test, self.prediction_random_forest)
                self.Score.append(self.random_forest_score)
            else:
                self.random_forest_score = roc_auc_score(y_test, self.prediction_random_forest , multi_class='ovo', average='weighted')
                self.Score.append(self.random_forest_score)

            Logger().log("Classification.py", "INFO", " Random Forest is performed on the Dataset ")
            with open('result/RandomForest_Model.pickle', 'wb') as f:
                pickle.dump(self.clf, f)
                Logger().log("Classification.py", "INFO",
                             "Saved RandomForest Model as RandomForest_Model.pickle")

            data.append(({
                'Accuracy': metrics.accuracy_score(y_test, self.prediction_random_forest),
                'Precision': metrics.precision_score(y_test, self.prediction_random_forest),
                'Recall or Sensitivity': metrics.recall_score(y_test, self.prediction_random_forest),
                'F1_Score': metrics.f1_score(y_test, self.prediction_random_forest),
                'roc_auc_score': roc_auc_score(y_test, self.prediction_random_forest, multi_class='ovo',
                                               average='weighted') ,
                "Model_Download": '<a href="//result/RandomForest_Model.pickle" download="RandomForest_Model.pickle">RandomForest_Model  </a>'

            }))

            ### XG BOOST

            self.xgb = XGBClassifier(objective='binary:logistic')
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(x_train, y_train.values.ravel())

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                                     n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(x_train, y_train.values.ravel())
            self.prediction_xgboost = self.xgb.predict(x_test)  # Predictions using the XGBoost Model

            self.Model.append(self.xgb)

            if len(y_test[
                       "output"].unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(y_test, self.prediction_xgboost)
                self.Score.append(self.xgboost_score)
            else:
                self.xgboost_score = roc_auc_score(y_test, self.prediction_xgboost , multi_class='ovo', average='weighted')  # AUC for XGBoost
                self.Score.append(self.xgboost_score)

            Logger().log("Classification.py", "INFO", " XGBoost is performed on the Dataset ")
            with open('result/XGBoost_Model.pickle', 'wb') as f:
                pickle.dump(self.xgb, f)
                Logger().log("Classification.py", "INFO",
                             "Saved XGBoost Model as XGBoost_Model.pickle")

            data.append(({
                'Accuracy': metrics.accuracy_score(y_test, self.prediction_xgboost),
                'Precision': metrics.precision_score(y_test, self.prediction_xgboost),
                'Recall or Sensitivity': metrics.recall_score(y_test, self.prediction_xgboost),
                'F1_Score': metrics.f1_score(y_test, self.prediction_xgboost),
                'roc_auc_score': roc_auc_score(y_test, self.prediction_xgboost, multi_class='ovo', average='weighted'),
                "Model_Download": '<a href="//result/XGBoost_Model.pickle" download="XGBoost_Model.pickle">XGBoost_Model  </a>'

            }))


            index = self.Score.index(max(self.Score))
            best_algo = self.L[index]
            best_Score = self.Score[index]
            best_Model = self.Model[index]

            recommended = []
            for i in self.L:
                if i == best_algo:
                    recommended.append("Highly Recommended")
                else:
                    recommended.append(" ")


            Logger().log("Classification.py", "INFO", str(self.L) + " "  + str(self.Score))

            Logger().log("Classification.py", "INFO", "Best_Model : " + str(best_algo) + "Best_Model_Score : " + str(best_Score))

            result = pd.DataFrame(data=data, columns=['Accuracy', 'Precision', 'Recall or Sensitivity',
                                                       'F1_Score', 'roc_auc_score' , "Model_Download" ],
                                   index=['Logistic_Regression', 'Random_Forest',
                                          'XG_Boost_Classifier'])


            results = pd.concat([result,pd.DataFrame(recommended,columns=["Recommended"],index=['Logistic_Regression', 'Random_Forest',
                                          'XG_Boost_Classifier'])],axis=1)


            return results
            #return best_algo, best_Score, best_Model, self.Score


        except Exception as e:
            Logger().log("Classification.py", "ERROR", "Error while performing Classification using Logistic , Random Forest , XGBoost"+ str(e))
            raise Exception("Error while performing Classification using Logistic , Random Forest , XGBoost "+ str(e))


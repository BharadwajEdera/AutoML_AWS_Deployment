import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
import pickle
from Logging.Logger import Logger
from sklearn import metrics


class Regression:
    def __init__(self):
        self.train = []
        self.test = []
    def fit(self,x_train, x_test, y_train, y_test):
        try:
            model = []
            data = []

            l = ['LinearRegression',
                 'lasso_reg',
                 'ridge_model',
                 'elasticnet_reg']

            # LinearRegression

            regression = LinearRegression()
            regression_model = regression.fit(x_train, y_train)
            model.append(regression_model)
            score_train = regression.score(x_train, y_train)
            self.train.append(score_train)

            score_test = regression.score(x_test, y_test)
            y_pred_Lreg = regression.predict(x_test)
            self.test.append(metrics.mean_squared_error(y_test, y_pred_Lreg))
            Logger().log("Regression.py", "INFO", " Linear Regression is performed on the Dataset ")

            with open('result/LinearRegression_Model.pickle', 'wb') as f:
                pickle.dump(regression, f)
            Logger().log("Regression.py", "INFO", " Linear Regression Model is saved as LinearRegression_Model.pickle")

            data.append(({
                "Score": score_test,
                'MSE': metrics.mean_squared_error(y_test, y_pred_Lreg),
                'RMSE': metrics.mean_squared_error(y_test, y_pred_Lreg, squared=False),
                'MAE': metrics.mean_absolute_error(y_test, y_pred_Lreg),
                'R-Squared': metrics.r2_score(y_test, y_pred_Lreg),
                "Model_Download": '<a href="//result/LinearRegression_Model.pickle" download="result/LinearRegression_Model.pickle">LinearRegression_Model  </a>'

            }))

            # Lasso Regularization

            lasscv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
            lasscv.fit(x_train, y_train)
            lasso_reg = Lasso(lasscv.alpha_)
            lasso_reg_model = lasso_reg.fit(x_train, y_train.values.ravel())
            model.append(lasso_reg_model)
            score_train = lasso_reg.score(x_train, y_train.values.ravel())
            self.train.append(score_train)

            score_test_Lasso = lasso_reg.score(x_test, y_test)
            y_pred_Lassoreg = lasso_reg_model.predict(x_test)
            self.test.append(metrics.mean_squared_error(y_test, y_pred_Lassoreg))
            Logger().log("Regression.py", "INFO", " Lasso Regression is performed on the Dataset ")

            with open('result/LassoRegression_Model.pickle', 'wb') as f:
                pickle.dump(lasso_reg_model, f)
            Logger().log("Regression.py", "INFO", " Lasso Regression Model is saved as LassoRegression_Model.pickle")

            data.append(({
                "Score": score_test_Lasso,
                'MSE': metrics.mean_squared_error(y_test, y_pred_Lassoreg),
                'RMSE': metrics.mean_squared_error(y_test, y_pred_Lassoreg, squared=False),
                'MAE': metrics.mean_absolute_error(y_test, y_pred_Lassoreg),
                'R-Squared': metrics.r2_score(y_test, y_pred_Lassoreg),
                "Model_Download": '<a href="//result/LassoRegression_Model.pickle" download="result/LassoRegression_Model.pickle">LassoRegression_Model  </a>'

            }))

            # Ridge
            ridgecv = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10, normalize=True)
            ridgecv.fit(x_train, y_train)
            ridge = Ridge(alpha=ridgecv.alpha_)
            ridge_model = ridge.fit(x_train, y_train.values.ravel())
            model.append(ridge_model)
            score_train = ridge_model.score(x_train, y_train.values.ravel())
            self.train.append(score_train)

            score_test_Ridge = ridge_model.score(x_test, y_test)
            y_pred_Ridgereg = ridge_model.predict(x_test)
            self.test.append(metrics.mean_squared_error(y_test, y_pred_Ridgereg))
            Logger().log("main.py", "Regression", " Ridge Regression is performed on the Dataset ")

            with open('result/RidgeRegression_Model.pickle', 'wb') as f:
                pickle.dump(ridge_model, f)
            Logger().log("main.py", "Regression", " Ridge Regression Model is saved as RidgeRegression_Model.pickle")
            data.append(({
                "Score": score_test_Ridge,
                'MSE': metrics.mean_squared_error(y_test, y_pred_Ridgereg),
                'RMSE': metrics.mean_squared_error(y_test, y_pred_Ridgereg, squared=False),
                'MAE': metrics.mean_absolute_error(y_test, y_pred_Ridgereg),
                'R-Squared': metrics.r2_score(y_test, y_pred_Ridgereg),
                "Model_Download": '<a href="//result/RidgeRegression_Model.pickle" download="result/RidgeRegression_Model.pickle">RideRegression_Model  </a>'

            }))

            # ElasticNet
            elasticCV = ElasticNetCV(alphas=None, cv=10)
            elasticCV.fit(x_train, y_train)
            elasticnet_reg = ElasticNet(alpha=elasticCV.alpha_, l1_ratio=0.5)
            elasticnet_reg_model = elasticnet_reg.fit(x_train, y_train.values.ravel())
            model.append(elasticnet_reg_model)
            score_train = elasticnet_reg.score(x_train, y_train.values.ravel())
            self.train.append(score_train)

            score_test_Elasticnetreg = elasticnet_reg.score(x_test, y_test)
            y_pred_Elasticnetreg = elasticnet_reg_model.predict(x_test)
            self.test.append(metrics.mean_squared_error(y_test, y_pred_Elasticnetreg))
            Logger().log("Regression.py", "INFO", " ElasticNet Regression is performed on the Dataset ")

            with open('result/ElasticNetRegression.pickle', 'wb') as f:
                pickle.dump(elasticnet_reg_model, f)
            Logger().log("Regression.py", "INFO", " ElasticNet Regression Model is saved as ElasticNetRegression_Model.pickle")

            data.append(({
                "Score": score_test_Elasticnetreg,
                'MSE': metrics.mean_squared_error(y_test, y_pred_Elasticnetreg),
                'RMSE': metrics.mean_squared_error(y_test, y_pred_Elasticnetreg, squared=False),
                'MAE': metrics.mean_absolute_error(y_test, y_pred_Elasticnetreg),
                'R-Squared': metrics.r2_score(y_test, y_pred_Elasticnetreg),
                "Model_Download": '<a href="//result/ElasticNetRegression.pickle" download="result/ElasticNetRegression.pickle">ElasticNetRegression  </a>'

            }))

            m = min(self.test)
            mix_index = self.test.index(m)
            best_algo = l[mix_index]

            recommended = []
            for i in l:
                if i == best_algo:
                    recommended.append("Highly Recommended")
                else:
                    recommended.append(" ")



            #Logger().log("Regression.py", "INFO", str(l) + str(self.test) )

            #Logger().log("Regression.py", "INFO", "Best_Model : " + str(l[max_index]) + "Best_Model_Score : " + str(self.test[max_index]))

            result = pd.DataFrame(data=data, columns=['Score', 'MSE', 'RMSE',
                                                      'MAE', 'R-Squared',"Model_Download"],
                                  index=['Linear_Regression', 'Lasso_Regression',
                                         'Ridge_Regression', 'ElasticNet_Regression'])

            results = pd.concat([result, pd.DataFrame(recommended, columns=["Recommended"],
                                                      index=['Linear_Regression', 'Lasso_Regression',
                                                             'Ridge_Regression', 'ElasticNet_Regression'])], axis=1)
            Logger().log("Regression.py", "INFO",
                         " Successfully created dataframe with performance measures , Model Download and Recommendations")
            return results

        except Exception as e:
            Logger().log("Regression.py", "ERROR",e)
            raise Exception("Error while performing Regression with Linear , Lasso , Ridge , ElasticNet"+ str(e))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


from Logging.Logger import Logger

class Scaling:
    def __init__(self):
        pass
    def Features(self,categorical_handled_df):
        try:
            if "output" in categorical_handled_df.columns:
                self.df = categorical_handled_df
                X = self.df.drop(["output"], axis=1)
                X_Columns = X.columns
                y = self.df[["output"]]
                scaler = StandardScaler()
                X_std = pd.DataFrame(scaler.fit_transform(X), columns=X_Columns)
                Logger().log("Feature_Scaling.py", "INFO",
                             "DataFrame Scaled with STANDARDIZATION (mean = 0 , variance = 1)")

                #min_max = MinMaxScaler()
                #X_std_MinMax = pd.DataFrame(min_max.fit_transform(X_std), columns=X_Columns)

                #Logger().log("Feature_Scaling.py", "INFO", "DataFrame Scaled with Normalization ")

                return X_std,y

            else:
                self.df = categorical_handled_df
                scaler = StandardScaler()
                X_std = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
                Logger().log("Feature_Scaling.py", "INFO",
                             "DataFrame Scaled with STANDARDIZATION (mean = 0 , variance = 1)")

                #min_max = MinMaxScaler()
                #X_std_MinMax = pd.DataFrame(min_max.fit_transform(X_std), columns=self.df.columns)
                #Logger().log("Feature_Scaling.py", "INFO", "DataFrame Scaled with Normalization ")

                return X_std, None


        except Exception as e:
            Logger().log("Feature_Scaling.py", "ERROR", "Failed to perform Feature scaling using Standardization"+ str(e))
            raise Exception("Failed to perform Feature scaling using Standardization"+ str(e))


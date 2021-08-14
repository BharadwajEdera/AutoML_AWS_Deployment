import pandas as pd
from Logging.Logger import Logger

class Categorical:
    def __init__(self):
        pass

    def Handle(self, missing_handled_df,type_of_problem):
        try:
            if "output" in missing_handled_df.columns:
                self.df = missing_handled_df.drop(["output"], axis=1)
                y = missing_handled_df["output"]
                data_types = self.df.dtypes
                categorical = data_types[data_types == "object"].index
                integer = data_types[data_types == "int64"].index
                for i in categorical:
                    if len(self.df[i].unique()) > 20:
                        self.df.drop([i], axis=1, inplace=True)
                Logger().log("Handle_Categorical.py", "INFO", "Removed object type unique Categories > 20")
                for i in integer:
                    if len(self.df[i].unique()) > 20:
                        self.df.drop([i], axis=1, inplace=True)
                Logger().log("Handle_Categorical.py", "INFO", "Removed int64 type unique Categories > 20")
                d = self.df.dtypes
                index = d[d == "object"].index

                if type_of_problem == "Regression":
                    for i in index:
                        ordinal_labels = missing_handled_df.groupby([i])['output'].mean().sort_values().index
                        ordinal_labels2 = {k: i for i, k in enumerate(ordinal_labels, 0)}
                        self.df[i] = self.df[i].map(ordinal_labels2)

                    output = pd.concat([self.df, y], axis=1)

                    Logger().log("Handle_Categorical.py", "INFO", "Handled object type Categories with TARGET GUIDED ENCODING")
                    return output

                elif type_of_problem == "Classification" :
                    for i in index:
                        d = self.df[i].value_counts().to_dict()
                        self.df[i] = self.df[i].map(d)

                    d = {k:v  for v,k in enumerate(y.unique())}
                    y = y.map(d)

                    output = pd.concat([self.df, y], axis=1)
                    Logger().log("Handle_Categorical.py", "INFO", "Handled object type Categories with Frequent categories imputation")
                    return output


            else:
                self.df = missing_handled_df
                data_types = self.df.dtypes
                categorical = data_types[data_types == "object"].index
                integer = data_types[data_types == "int64"].index
                for i in categorical:
                    if len(self.df[i].unique()) > 20:
                        # print(i, len(self.df[i].unique()))
                        self.df.drop([i], axis=1, inplace=True)
                Logger().log("Handle_Categorical.py", "INFO", "Removed object type unique Categories > 20")

                for i in integer:
                    if len(self.df[i].unique()) > 20:
                        self.df.drop([i], axis=1, inplace=True)
                Logger().log("Handle_Categorical.py", "INFO", "Removed int64 type unique Categories > 20")

                d = self.df.dtypes
                index = d[d == "object"].index
                for i in index:
                    d = self.df[i].value_counts().to_dict()
                    self.df[i] = self.df[i].map(d)


                output = self.df

                Logger().log("Handle_Categorical.py", "INFO",
                             "Handled object type Categories with Frequent categories imputation")

                return output

        except Exception as e:
            Logger().log("Handle_Categorical.py", "ERROR", "Failed to Handle Categorical values by Target Guided Encoding or Count Frequency Imputation"+ str(e))
            raise Exception("Failed to Handle Categorical values by Target Guided Encoding or Count Frequency Imputation"+ str(e))

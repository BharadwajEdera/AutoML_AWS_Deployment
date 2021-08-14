from Logging.Logger import Logger

class Outliers:
    def __init__(self):
        pass
    def Handle(self,categorical_handled_df):
        try:
            self.df = categorical_handled_df
            for Column_Name in self.df.columns:
                IQR = self.df[Column_Name].quantile(0.75) - self.df[Column_Name].quantile(0.25)
                lower_boundary = self.df[Column_Name].quantile(0.25) - (1.5 * IQR)
                upper_boundary = self.df[Column_Name].quantile(0.75) + (1.5 * IQR)

                self.df.loc[self.df[Column_Name] >= upper_boundary, Column_Name] = upper_boundary
                self.df.loc[self.df[Column_Name] <= lower_boundary, Column_Name] = lower_boundary

            Logger().log("Outliers.py", "INFO", "Outlier removed Successfully with IQR,LB,UB")

            return self.df
        except Exception as e:
            Logger().log("Outliers.py", "ERROR", "Unable to handle Outlier"+ str(e))
            raise Exception("Unable to handle Outlier"+ str(e))


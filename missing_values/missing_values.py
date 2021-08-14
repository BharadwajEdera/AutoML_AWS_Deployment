

import numpy as np
from Logging.Logger import Logger


class check_missing_values:
    def __init__(self):
        pass
    def handle_missing(self,df):

        try:
            self.df1 = df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
            Logger().log("missing_values.py", "INFO", "Missing values handled Successfully with RANDOM SAMPLE IMPUTATION")
            return self.df1
        except Exception as e:
            Logger().log("missing_values.py", "ERROR", "Failed to handle missing Values"+ str(e))
            raise Exception("Failed to handle missing Values with Random Sample Imputation"+ str(e))





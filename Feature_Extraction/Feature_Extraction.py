from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pickle
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf
import plotly.graph_objects as go
py.offline.init_notebook_mode(connected = True)
cf.go_offline()
import plotly.express as px
import numpy as np
import pandas as pd


from Logging.Logger import Logger


class PCA_dimensionality:
    def __init__(self):
        pass

    def PCA_fit_transform(self, X, y):
        try:
            p = PCA()
            new_data = p.fit_transform(X)


            fig = px.line(x=range(1, len(X.columns) + 1), y=np.cumsum(p.explained_variance_ratio_),
                          labels={'x': "Number of Components", 'y': "Explained Variance Ratio"},
                          title="PCA : The KNEE Method")

            # finding the value Optimal Value of KNEE

            kn = KneeLocator(range(1, len(X.columns) + 1), np.cumsum(p.explained_variance_ratio_), curve='concave',
                             direction='increasing')
            knee = kn.knee
            print(knee)

            fig.add_vrect(x0=knee, x1=knee)
            fig.write_image("result/PCA_KNEE_Plotly.png")
            fig.show()

            pca = PCA(n_components=knee)
            PCA_data = pca.fit_transform(X)
            data = pd.DataFrame(PCA_data)
            Logger().log(f"Feature_Extraction.py", "INFO", "Number of Columns Reduced From " + str(len(X.columns)) + " to " + str(len(data.columns)) + " using PCA with KNEE =" +  str(knee))
            return data , y
        except Exception as e:
            Logger().log(f"Feature_Extraction.py", "INFO", "Error while performing Feature Extraction with PCA"+ str(e))
            raise Exception("Error while performing Feature Extraction with PCA"+ str(e))



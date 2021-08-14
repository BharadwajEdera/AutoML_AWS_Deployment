### Libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from wsgiref import simple_server
from flask import Response
import os
from flask_cors import CORS, cross_origin
import os
import glob
import re

### Files
from missing_values.missing_values import check_missing_values
from Logging.Logger import Logger
from Handling_Categorical.Handling_Categorical import Categorical
from Feature_Scaling.Feature_Scaling import Scaling
from Outliers.Outliers import Outliers
from Feature_Extraction.Feature_Extraction import PCA_dimensionality
from Regression.Regression import Regression
from Classification.Classification import Classification
from Clustering.Clustering import Clustering
from Type_of_Dataset.Type_of_Dataset import Dataset
from DataBase.DataBase import database

application = Flask(__name__)



@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@application.route('/AutoML', methods=['POST','GET']) # for calling the API from Postman/SOAPUI
@cross_origin()
def AutoML():
    if (request.method=='POST'):

        try:
            type_of_problem = request.form['type_of_problem']
            type_of_dataset = request.form['type_of_dataset']
            output_column = request.form['output_column']
            dataset_path = request.form['dataset_path']

            Logger().log("main.py", "INFO", "                                                      ")
            Logger().log("main.py", "INFO", "*******----------*******######******-------------******")
            Logger().log("main.py", "INFO", "*******----------NEW REQUEST STARTED :: " + str(type_of_problem) + "-------------******")
            Logger().log("main.py", "INFO", "--------Received Input from User-------------")
        except:
            Logger().log("main.py", "ERROR", "Error while reading the input from user")
            return "Error while reading the input from user"

        if output_column in ["None" or '']:
            output_column = None



        try:
            dir = os.getcwd() + "/result"
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
            Logger().log("main.py", "INFO", "Deleted all Existing Old Files in the result Directory")
        except Exception as e:
            Logger().log("main.py", "INFO", "Error while deleting the Existing Old Files in the result Directory"  + str(e))
            return "There is no directory named as result. Error while deleting the Existing Old Files in the result Directory" + str(e)

        try:
            df = Dataset().DataFrame(type_of_dataset, output_column, dataset_path)
            Logger().log("main.py", "INFO", str(type_of_dataset) + " Dataset reading successful")
        except Exception as e:
            Logger().log("main.py", "ERROR", str(type_of_dataset) + " Dataset reading Failed" + str(e))
            return "Error while reading dataset. Either of the Reason Listed Below \n 1.Enter Dataset Path along with ." + str(type_of_dataset) + "  extension \n2.Check Spelling of " + str(type_of_dataset) + " entered \n 3.Entered file may be in other File Format \n 4.Entered " + str(type_of_dataset) + " File doesn't exists" + str(e)

        try:
            l = []
            pattern = re.compile("[^0-9a-zA-Z]+")
            for i in df.columns:
                l.append(pattern.sub("", i))
            df.columns = l
            Logger().log("main.py", "INFO", " Removed  Special Characters in Column names of Dataset ")

            df.to_csv("result/Train.csv",index=False)
            Logger().log("main.py", "INFO", " Saved dataset into result directory as Train.csv ")

        except Exception as e:
            Logger().log("main.py", "ERROR", "Error : Either of the Reasons Listed below : \n 1.Regular Expressions Failed to remove special characters from DataFrame Columns \n 2.Unable to save Train.csv into result.directory " + str(e))
            return "Error : Either of the Reasons Listed below : \n 1.Regular Expressions Failed to remove special characters from DataFrame Columns \n 2.Unable to save Train.csv into result.directory " + str(e)

        try:
            df = database("result/Train.csv").database_dataframe(type_of_problem)
            Logger().log("main.py", "INFO", " The dataset is successfully Loaded into Cassandra Database ")
            Logger().log("main.py", "INFO", " The Table from Cassandra Database is successfully Loaded into Dataframe  ")
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error in connecting to cassandra Database"+str(e))
            return "Error in connecting to cassandra Database. \n Following might be the reason \n" + str(e) + "\n No need to worry." + "\n Check your Internet Connection" +"\n  Try to startover and resend your request"

        try:
            missing_handled_df = check_missing_values().handle_missing(df)
        except Exception as e:
            Logger().log("main.py", "ERROR", "Failed to handle missing Values with Random Sample Imputation"+str(e))
            return "Failed to handle missing Values with Random Sample Imputation" +str(e)

        try:
            categorical_handled_df = Categorical().Handle(missing_handled_df,type_of_problem)
        except Exception as e:
            Logger().log("main.py", "ERROR", "Failed to handle missing Values with Random Sample Imputation" +str(e))
            return "Failed to handle missing Values with Random Sample Imputation" +str(e)
        try:
            outliers_handled_df = Outliers().Handle(categorical_handled_df)
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while handling Outliers " + str(e))
            return "Error while handling Outliers \n" + str(e)

        try:
            X_scaled,y = Scaling().Features(outliers_handled_df)
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while performing Scaling " + str(e))
            return "Error while performing Scaling \n" + str(e)

        try:
            PCA_X , y = PCA_dimensionality().PCA_fit_transform(X_scaled,y)
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while performing Feature Extraction with PCA " + str(e))
            return "Error while performing Feature Extraction with PCA \n" + str(e)

        try:
            if type_of_problem == "Regression":
                x_train, x_test, y_train, y_test = train_test_split(PCA_X, y, test_size=0.25, random_state=355)
                Logger().log("main.py", "INFO", "train_test_split is successful with Train : 75 % , Test : 25 % ")
                result = Regression().fit( x_train, x_test, y_train, y_test)
                html = result.to_html(escape=False)
                return html
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while performing Regression " + str(e))
            return "Error while performing Regression \n" + str(e)

        try:
            if type_of_problem == "Classification":
                x_train, x_test, y_train, y_test = train_test_split(PCA_X, y, test_size=0.25, random_state=355)
                Logger().log("main.py", "INFO", "train_test_split is successful with Train : 75 % , Test : 25 % ")
                #best_algo, best_Score, best_Model, Score_List = Classification().fit(x_train, x_test, y_train, y_test)
                result = Classification().fit(x_train, x_test, y_train, y_test)
                html = result.to_html(escape=False)
                return html
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while performing Classification " + str(e))
            return "Error while performing Classification \n" + str(e)

        try:
            if type_of_problem == "Clustering":
                x_train, x_test = train_test_split(PCA_X, test_size=0.30, random_state=355)
                Logger().log("main.py", "INFO", "train_test_split is successful with Train : 75 % , Test : 25 % ")
                result = Clustering().fit(x_train, x_test)
                html = result.to_html(escape=False)
                return html
        except Exception as e:
            Logger().log("main.py", "ERROR", "Error while performing Clustering " + str(e))
            return "Error while performing Clustering \n" + str(e)

        Logger().log("main.py", "INFO", "REQUEST ENDED")
        Logger().log("main.py", "INFO", "                                                      ")
        Logger().log("main.py", "INFO", "*******----------*******######******-------------******")






if __name__ == '__main__':
    application.run()









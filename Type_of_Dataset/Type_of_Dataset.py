import pandas as pd
from flask import Flask, render_template, request, jsonify
import json
from Logging.Logger import Logger


class Dataset:
    def __init__(self):
        pass

    def DataFrame(self, type_of_dataset, output_column, dataset_path):
        try:
            if type_of_dataset == "csv":
                try:
                    df = pd.read_csv(dataset_path)
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading csv file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading csv file"+ str(e))
                    raise Exception("ERROR while reading csv file"+ str(e))


            elif type_of_dataset == "xlsx":
                try:
                    df = pd.read_excel(dataset_path , engine='openpyxl' )
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading xlsx file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading xlsx file" + str(e))
                    raise Exception("ERROR while reading xlsx file"+ str(e))



            elif type_of_dataset == "json":
                try:
                    with open(dataset_path) as f:
                        data = json.load(f)

                    df = pd.DataFrame(data)
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading json file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading json file"+ str(e))
                    raise Exception("ERROR while reading json file"+ str(e))


            elif type_of_dataset == "tsv":
                try:
                    df = pd.read_table(dataset_path, sep='\t')
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading tsv file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading tsv file" + str(e))
                    raise Exception("ERROR while reading tsv file"+ str(e))


            elif type_of_dataset == "html":
                try:
                    df = pd.read_html(dataset_path)
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading html file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading html file"+ str(e))
                    raise Exception("ERROR while reading html file"+ str(e))


            elif type_of_dataset == "txt":
                try:
                    df = pd.read_csv(dataset_path, sep=" ")
                    df.rename(columns={output_column: 'output'}, inplace=True)
                    Logger().log("Type_of_Dataset.py", "INFO", "Reading txt file successful")
                    return df
                except Exception as e:
                    Logger().log("Type_of_Dataset.py", "ERROR", "ERROR while reading txt file"+ str(e))
                    raise Exception("ERROR while reading tsv file"+ str(e))


        except Exception as e:
            Logger().log("Type_of_Dataset.py", "ERROR", e)
            raise Exception("ERROR while reading file" + str(e))

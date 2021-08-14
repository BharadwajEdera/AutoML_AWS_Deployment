from cassandra.util import OrderedMapSerializedKey
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
import csv
from tqdm import tqdm
from Logging.Logger import Logger

class database:

    def __init__(self, csv_file):
        try:
            cloud_config = {'secure_connect_bundle': 'secure-connect-automldb.zip'}
            Logger().log("DataBase.py", "INFO", " Successfully configured  'secure-connect-automldb.zip' in cassandra database ")
            auth_provider = PlainTextAuthProvider('QTHrHUiIpBqCzdEARCwfPAyN',
                                                  'MIMI10nlWagbX5hjZGBheHm+vDrnSYhCOClA8+StvdLM2gw_TRu5UZRmuDlU7zKwBjvqwTzt0WIA-DfzHfD8NSlCc+HzEzl9w2zCoQABEenf9g+AKD_NhSa44vnNYovE')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            Logger().log("DataBase.py", "INFO", " Successfully created Cluster in cassandra Database")
            self.session = cluster.connect('AutoMLKS')
            Logger().log("DataBase.py", "INFO", " Successfully connected to Cluster in cassandra Database")
            self.session.default_timeout = 60
            self.df1 = pd.read_csv(csv_file)
            self.csv_data = csv.reader(open(csv_file))
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed to connected to Cluster in cassandra Database" + str(e))
            raise Exception("Failed to connected to Cluster in cassandra Database"+ str(e))


    def existing_tables(self):
        try:
            row = self.session.execute("SELECT table_name FROM system_schema.tables WHERE keyspace_name='AutoMLKS';").all()
            Logger().log("DataBase.py", "INFO", " Successful in checking the existing tables in cassandra database")
            return row
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed in checking the existing tables in cassandra database"+ str(e))
            raise Exception("Failed in checking the existing tables in cassandra database"+ str(e))

    def is_table_exists(self, table_name):
        try:
            row = self.existing_tables()
            for i in range(len(row)):
                if str(row[i][0]) == table_name:
                    Logger().log("DataBase.py", "INFO", "Table with the given table name already Exists in Database")
                    return True
                else:
                    continue
            Logger().log("DataBase.py", "INFO", "Table with the given table name doesn't Exists in Database")
            return False
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed in checking the particular table in Database"+ str(e))
            raise Exception(" Failed in checking the particular table in Database"+ str(e))


    def drop_table(self, table_name):
        try:
            self.session.execute("DROP table " + str(table_name) + ";").all()
            Logger().log("DataBase.py", "INFO", "Successfully deleted the Table")
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed in checking the particular table in Database"+ str(e))
            raise Exception(" Failed to delete table in Database"+ str(e))



    def create_table(self, table_name):
        try:
            s = ''
            for i in self.df1.columns:
                s += str(i) + ' ' + 'text' + ','
            print("CREATE TABLE " + str(table_name) + " (" + str(s) + "PRIMARY KEY (" + str(self.df1.columns[0]) + ") );")
            self.session.execute("CREATE TABLE " + str(table_name) + " (" + str(s) + "PRIMARY KEY (" + str(
                self.df1.columns[0]) + ") );").one()
            Logger().log("DataBase.py", "INFO", "Successfully Created new table in Cassandra Database")
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed to create new table in Cassandra Database"+ str(e))
            raise Exception(" Failed to create new table in Cassandra Database"+ str(e))


    def insert_into_table(self, table_name):
        try:
            s1 = ''
            for i in self.df1.columns:
                s1 += str(i) + ','

            s2 = '%s,' * len(self.df1.columns)

            header = next(self.csv_data)
            for i in tqdm(self.csv_data):
                row = self.session.execute(
                    "INSERT INTO " + str(table_name) + " (" + str(s1[:-1]) + ") VALUES (" + str(s2[:-1]) + ");", i).one()
            Logger().log("DataBase.py", "INFO", "Successfully Inserted data into Cassandra Database")
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed to Insert Data into Cassandra Database"+ str(e))
            raise Exception(" Failed to Insert Data into Cassandra Database"+ str(e))

    def database_dataframe(self, type_of_problem):
        try:
            type_of_problem = type_of_problem.lower()
            table_name = str(type_of_problem) + str(self.df1.shape[0])
            print(table_name)
            if self.is_table_exists(table_name):
                self.drop_table(table_name)
                self.create_table(table_name)
                self.insert_into_table(table_name)
                Logger().log("DataBase.py", "INFO", " Successful in DataFrame to Database ")
                return self.show_table(table_name)
            else:
                self.create_table(table_name)
                self.insert_into_table(table_name)
                Logger().log("DataBase.py", "INFO", " Successful in DataFrame to Database ")
                return self.show_table(table_name)

        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " DataFrame to Database"+ str(e))
            raise Exception(" Failed in DataFrame to Database"+ str(e))




    def pandas_factory(self, colnames, rows):
        try:
            # Convert tuple items of 'rows' into list (elements of tuples cannot be replaced)
            rows = [list(i) for i in rows]

            # Convert only 'OrderedMapSerializedKey' type list elements into dict
            for idx_row, i_row in enumerate(rows):

                for idx_value, i_value in enumerate(i_row):

                    if type(i_value) is OrderedMapSerializedKey:
                        rows[idx_row][idx_value] = dict(rows[idx_row][idx_value])
            Logger().log("DataBase.py", "INFO", "Successfully converted the Database dataframe columns to same as Entered dataframe columns ")
            return pd.DataFrame(rows, columns=colnames)
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed to convert the Database dataframe columns to same as Entered dataframe columns"+ str(e))
            raise Exception("  Failed to convert the Database dataframe columns to same as Entered database columns"+ str(e))



    def show_table(self, table_name):
        try:
            colnames = self.df1.columns
            rows = self.session.execute("SELECT * FROM " + str(table_name) + ";").all()
            df = self.pandas_factory(colnames, rows)
            # for i in range(len(self.df1.dtypes)):
            # self.df[self.df.columns[i]] = self.df[self.df.columns[i]].astype(self.df1.dtypes[i])
            Logger().log("DataBase.py", "INFO", "Successfully converted data in database to Dataframe ")
            return self.df1
        except Exception as e:
            Logger().log("DataBase.py", "ERROR", " Failed to convert data in database to Dataframe"+ str(e))
            raise Exception("  Failed to convert data in database to Dataframe"+ str(e))

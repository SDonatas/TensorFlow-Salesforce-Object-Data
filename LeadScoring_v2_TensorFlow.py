#Application name Lead Oracle 2.0
#Using TensorFlow machine learning library
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

#IMPORT LIBRARIES
import os
from simple_salesforce import Salesforce
from datetime import datetime
from datetime import timedelta
from time import sleep
#from collections import Counter

#Machine Learning Libraries
import tensorflow as tf
from tensorflow import keras

#Other data libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import math
import logging
import math


#Path variable
try:
    script_path = os.path.dirname(os.path.abspath(__file__)) + "/"
except:
    script_path = ""

#Logging
logging.basicConfig(filename=script_path + 'LeadScoring_FT_TensorFlow_Conversions.log',level=logging.DEBUG)
logging.info(tf.__version__)
logging.info("Script path: " + script_path)

#Locate settings
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

#******************SETTINGS******************************************************************************************

#NOTE TAKE CARE WHEN HOSTING THE SCRIPT, AS CREDENTIALS ARE NAKED
#salesforce credentials
username=''
password=''
organizationId=''

#salesforce query global settings
sitename = "FT"
daynumber = 365 * 2.5
exlude_new_days = 0

#Loop settings
loop_sleep_seconds = 3600
reevaluate_model_after_days = 30
loop_timestamp_query_days = 5

Query = "SELECT id, CreatedDate, Site__c, ConvertedOpportunityId, IsConverted, Session_count__c, Lead_score_change__c, Lead_score_timestamp__c, Lead_score_timestamp_2__c, Rating__c, IP_city__c, Country, Destination__c, Device__c, Division__c, Created_weekday__c, Last_session_source__c, LeadSource, Dates_since_sign_up__c, Email_CTOR__c, Email_Open__c, Engaged_with_chat__c, Engaged_with_form__c, HasOptedOutOfEmail, Have_you_been_to_this_destination_before__c, Last_session_min__c, Lead_profile_updated__c, Lead_survey_updated__c, New_visitor__c, Pages_last_session__c, Read_reviews__c, Sign_up_source__c, TotalEmailsOpened__c, TotalEmailsSent__c, TotalTimesOpened__c, TotalUniqueClicks__c, Total_pages__c, Total_time_min__c, utm_source__c, utm_term__c, RecordTypeId, Email, Price_per_day__c, status FROM Lead WHERE Site__c = '"+str(sitename) +"' AND CreatedDate > " + str(datetime.now() - timedelta(days=daynumber))[:10] + "T00:00:00Z" + " AND CreatedDate <= " + str(datetime.now() - timedelta(days=exlude_new_days))[:10] + "T23:59:59Z" + " ORDER BY CreatedDate DESC"
Query_opp = "SELECT id, IsWon, isLost__c from Opportunity WHERE Site__c = '"+str(sitename) +"' AND CreatedDate > " + str(datetime.now() - timedelta(days=daynumber))[:10] + "T00:00:00Z" + "AND id <> Null ORDER BY CreatedDate DESC"
Query_loop = "SELECT id, CreatedDate, Site__c, ConvertedOpportunityId, IsConverted, Session_count__c, Lead_score_change__c, Lead_score_timestamp__c, Lead_score_timestamp_2__c, Rating__c, IP_city__c, Country, Destination__c, Device__c, Division__c, Created_weekday__c, Last_session_source__c, LeadSource, Dates_since_sign_up__c, Email_CTOR__c, Email_Open__c, Engaged_with_chat__c, Engaged_with_form__c, HasOptedOutOfEmail, Have_you_been_to_this_destination_before__c, Last_session_min__c, Lead_profile_updated__c, Lead_survey_updated__c, New_visitor__c, Pages_last_session__c, Read_reviews__c, Sign_up_source__c, TotalEmailsOpened__c, TotalEmailsSent__c, TotalTimesOpened__c, TotalUniqueClicks__c, Total_pages__c, Total_time_min__c, utm_source__c, utm_term__c, RecordTypeId, Email, Price_per_day__c, status FROM Lead WHERE Site__c = '"+str(sitename) +"' AND CreatedDate > " + str(datetime.now() - timedelta(days=daynumber))[:10] + "T00:00:00Z" + " AND CreatedDate <= " + str(datetime.now() - timedelta(days=exlude_new_days))[:10] + "T23:59:59Z" +" AND Lead_score_timestamp__c >= "+str(datetime.now() - timedelta(days=loop_timestamp_query_days))[:10] + "T23:59:59Z"+" AND IsConverted = False ORDER BY CreatedDate DESC"


#*************  Mapping settings & Outlier specifications ***************************
#Please note bellow mapping fields if eddited need to be edited within model also (only MapToNumbers, not Nomapping)

MapToNumbers = ['Device__c',
#'Destination__c', 
'Country',
'IP_city__c',                
'Created_weekday__c', 
'Last_session_source__c', 
'LeadSource', 
'HasOptedOutOfEmail',
'Have_you_been_to_this_destination_before__c', 
'Sign_up_source__c',
'utm_source__c',
'utm_term__c',
#'RecordTypeId'
               ] 

NoMapping= [
'Email_Open__c', 
'Last_session_min__c', 
'Pages_last_session__c',
'TotalEmailsOpened__c',
'TotalEmailsSent__c', 
'TotalTimesOpened__c', 
'TotalUniqueClicks__c', 
'Total_pages__c', 
'Total_time_min__c',
'Read_reviews__c',
'New_visitor__c',
'Engaged_with_chat__c',
'Engaged_with_form__c',
'Price_per_day__c'
] 

#************************FUNCTIONS AND CLASES**********************************************************

class LeadScoring:
    def __init__(self, username, password, organizationId):
        logging.debug("Initiating Salesforce object...")
        self.sf = Salesforce(username=username, password=password, organizationId=organizationId)
        logging.debug("Initiating Salesforce object... Done")
        self.data = None
        self.ObjectCreationDateTimestamp = datetime.now()
    
    def QueryData(self, Query, Query_opp=Query_opp, stage="initial", sitename = sitename, daynumber = daynumber, exlude_new_days=exlude_new_days):
        
        assert stage == "initial" or stage == "loop"
        logging.info("Running queries...")
        logging.debug("Running Lead query...")
        self.data = self.sf.query_all(Query)
        logging.debug("Running Lead query... Done")
        
        if stage == "initial":
            logging.debug("Running Opportunity query...")
            self.data_opp = self.sf.query_all(Query_opp)
            logging.debug("Running Opportunity query... Done")
            
        logging.info("Running queries... Done")
        
        if len(self.data["records"]) == 0:
            logging.info("Query empty, exiting function...")
            return
        
        assert len(self.data["records"]) > 0
        self.data = [[item for item in self.data["records"][0].keys()][1:]] + [filtered_row[1:] for filtered_row in [[row for row in item.values()] for item in self.data["records"]]]
        
        if stage == 'initial':
            assert len(self.data_opp["records"]) > 0
            self.data_opp = [[item for item in self.data_opp["records"][0].keys()][1:]] + [filtered_row[1:] for filtered_row in [[row for row in item.values()] for item in self.data_opp["records"]]]

        #numpy.isnan(number)
        
        #Try to Convert to floats string numbers
        def process_data(data_to_process):
            logging.info("Processing SF Query Data...")
            for Y_key, y_item in enumerate(data_to_process):
                for x_key, x_item in enumerate(y_item):
                    
                    #Process none types
                    if x_item == None:
                        data_to_process[Y_key][x_key] = "NO_DATA"
                    
                    #Process NaN foats
                    elif type(x_item) == float:
                        if math.isnan(x_item) == True:
                            data_to_process[Y_key][x_key] = "NO_DATA"
                        else:
                            pass
                    #Process booleans, i.e. leave as it is
                    elif type(x_item) == bool:
                        if x_item == True:
                            data_to_process[Y_key][x_key] = 1.0
                        else:
                            data_to_process[Y_key][x_key] = 0.0
                    
                    #Process integers
                    elif type(x_item) == int:
                        data_to_process[Y_key][x_key] = float(data_to_process[Y_key][x_key])
                    
                    #Process emtry strings
                    elif x_item.strip() == "" or x_item.strip() == " ":
                        data_to_process[Y_key][x_key] = "NO_DATA"
                    
                    elif x_item.isdigit():
                        data_to_process[Y_key][x_key] = float(data_to_process[Y_key][x_key])
            logging.info("Processing SF Query Data... Done")
            return data_to_process
        
        self.data = process_data(self.data)
        self.data = pd.DataFrame(self.data[1:], columns=self.data[0])
        #self.data.to_csv("original_testing_v2.csv")
       
        if stage == 'initial': 
            #Make opportunity object and later join to main leads
            self.leads = pd.DataFrame(self.data[(self.data["IsConverted"] == 0) & (self.data["Lead_score_timestamp__c"] != self.data["Lead_score_timestamp_2__c"])])
            self.data_opp = process_data(self.data_opp)
            self.data_opp = pd.DataFrame(self.data_opp[1:], columns=self.data_opp[0])
                    
            #Replace opportunity ID with outcome
            if "ConvertedOpportunityId" in list(self.data.columns.values):
                self.data = self.data.join(self.data_opp.set_index('Id'), on='ConvertedOpportunityId')
                #self.data['outcome'] = np.where(self.data['IsWon'] == True, 1, 0)
                self.data['outcome'] = np.where(self.data["IsConverted"] == 1, 1, 0)
                self.data = self.data[((self.data["IsConverted"] == 0) & ((self.data["Status"] == "Unqualified") | (self.data["Status"] == "Not converted"))) | (self.data["IsConverted"] == 1)]
            else:
                raise ValueError("please add field into Leads table: ConvertedOpportunityId")
        elif stage == 'loop':
            self.leads = pd.DataFrame(self.data[(self.data["IsConverted"] == 0) & (self.data["Lead_score_timestamp__c"] != self.data["Lead_score_timestamp_2__c"])])
            delattr(self, 'data')
        else:
            raise ValueError("Can not recognise operation stage")
            #Testing save
            #self.leads = pd.DataFrame(self.data)
            #self.leads.to_csv("Leads_original_3.csv")

            #Additional filtering layer to get only with known outcomes
            #self.data = self.data[(self.data["IsWon"] == True) | (self.data["isLost__c"] == True)]
            #self.data = self.data[((self.data["IsConverted"] == 0) & ((self.data["Status"] == "Unqualified") | (self.data["Status"] == "Not converted"))) | (self.data["IsConverted"] == 1)]
            #self.data.to_csv("original.csv")
            #status
    
    def Reframe_data_optional(self, stage="initial"):
        logging.info("Reframming data fields...")
        assert stage == "initial" or stage == "loop" or stage == 'All_leads'
        #This step is to deal with too many diverse labels for certain fields such as city
        #This step will make sure only most frequent are being kept, while others simply labeled as NO_DATA
        
        #'IP_city__c'
        
        def field_reframe(Reframe_field, Data_object, field, sample_size):
            logging.debug("Reframing field {} ...".format(str(field)))
            field_index = list(Data_object.groupby([field])['outcome'].sum().sort_values(ascending=False).head(sample_size).index.values)
            #field_values = list(Data_object.groupby([field])['outcome'].sum().sort_values(ascending=False).head(sample_size).values)

            for key, index in enumerate(field_index):
                Reframe_field[index] = index
            
            ItemsToReframe = {x:"NO_DATA" for x in list(Data_object[field].values) if x not in list(Reframe_field.keys())}
            logging.debug("Items to reframe: {}".format(str(len(ItemsToReframe))))
            if len(ItemsToReframe) > 0:
                Data_object[field] = Data_object[field].replace(ItemsToReframe)
            else:
                logging.debug("Skipping reframing, no items to reframe, i.e. specified unique top items to keep fits all unique combinations")
            
            logging.debug("Reframing field {} ...".format(str(field)) + "Done")
            return Reframe_field, Data_object
        
        def utm_term_reframe_separate(Reframe_field, Data_object, field):
            logging.debug("Reframing field {} ...".format(str(field)))
            field_index = list(Data_object["utm_term__c"].values)
            #list(Data_object[(Data_object['utm_term__c'].str.contains("tour")) | (Data_object['utm_term__c'].str.contains("cruise")) | (Data_object['utm_term__c'].str.contains("vacation")) | (Data_object['utm_term__c'].str.contains("trip")) | (Data_object['utm_term__c'].str.contains("travel"))]["utm_term__c"].values)
            
            for key, item in enumerate(field_index):
                if "tour" in item:
                    Reframe_field[item] = "tour"
                elif "cruise" in item:
                    Reframe_field[item] = "cruise"
                elif "vacation" in item:
                    Reframe_field[item] = "vacation"
                elif "trip" in item:
                    Reframe_field[item] = "trip"
                elif "travel" in item:
                    Reframe_field[item] = "travel"
                else:
                    Reframe_field[item] = "NO_DATA"
                
            Data_object[field] = Data_object[field].replace(Reframe_field)
            logging.debug("Reframing field {} ...".format(str(field)) + "Done")
            return Reframe_field, Data_object
        
        
        if stage == "initial":
            self.Reframe_field = dict()
            self.Reframe_field['IP_city__c'] = dict()
            self.Reframe_field['IP_city__c'], self.data = field_reframe(self.Reframe_field['IP_city__c'], self.data, 'IP_city__c', 100)
            
            self.Reframe_field['Country'] = dict()
            self.Reframe_field['Country'], self.data = field_reframe(self.Reframe_field['Country'], self.data, 'Country', 100)
            
            self.Reframe_field['utm_term__c'] = dict()
            self.Reframe_field['utm_term__c'], self.data = utm_term_reframe_separate(self.Reframe_field['utm_term__c'], self.data, 'utm_term__c')
           
            #city_index = list(self.data.groupby(['IP_city__c'])['outcome'].sum().sort_values(ascending=False).head(100).index.values)
            #city_values = list(self.data.groupby(['IP_city__c'])['outcome'].sum().sort_values(ascending=False).head(100).values)
            #Reframe_IP_city__c = dict()
            #for key, index in enumerate(city_index):
            #    Reframe_IP_city__c[index] = city_values[key]

            #ItemsToReframe = {x:"NO_DATA" for x in list(self.data["IP_city__c"].values) if x not in list(Reframe_IP_city__c.keys())}
            #self.data['IP_city__c'] = self.data['IP_city__c'].replace(ItemsToReframe) 
        
        if stage == "All_leads" or stage == 'loop':
            if self.leads.shape[0] == 0:
                logging.info("No leads to score, exiting function")
                return
            
            ItemsToReframe = {x:"NO_DATA" for x in list(self.leads['IP_city__c'].values) if x not in list(self.Reframe_field['IP_city__c'].keys())}
            self.leads['IP_city__c'] = self.leads['IP_city__c'].replace(ItemsToReframe)
            
            ItemsToReframe = {x:"NO_DATA" for x in list(self.leads['Country'].values) if x not in list(self.Reframe_field['Country'].keys())}
            self.leads['Country'] = self.leads['Country'].replace(ItemsToReframe)
            
            self.Reframe_field['utm_term__c'] = dict()
            self.Reframe_field['utm_term__c'], self.leads = utm_term_reframe_separate(self.Reframe_field['utm_term__c'], self.leads, 'utm_term__c')
           
            
            
            #ItemsToReframe = {x:"NO_DATA" for x in list(self.leads['utm_term__c'].values) if x not in list(self.Reframe_field['utm_term__c'].keys())}
            #self.leads['utm_term__c'] = self.leads['utm_term__c'].replace(ItemsToReframe)
            
            
            
            #utm_term__c
            #city_index = list(self.leads.groupby(['IP_city__c'])['outcome'].sum().sort_values(ascending=False).head(100).index.values)
            #city_values = list(self.leads.groupby(['IP_city__c'])['outcome'].sum().sort_values(ascending=False).head(100).values)
            #Reframe_IP_city__c = dict()
            #for key, index in enumerate(city_index):
            #    Reframe_IP_city__c[index] = city_values[key]

            #ItemsToReframe = {x:"NO_DATA" for x in list(self.leads["IP_city__c"].values) if x not in list(Reframe_IP_city__c.keys())}
            #self.leads['IP_city__c'] = self.leads['IP_city__c'].replace(ItemsToReframe) 
        
        logging.info("Reframming data fields... Done")
                    
    def mapfields(self, column_list_to_map, numeric_column_list_to_use, labelfields, stage="initial"):
        
        assert stage == "initial" or stage == "loop" or stage == 'All_leads'
        
        logging.info("Mapping fields for {} stage".format(stage))
        
        if stage == "initial":
            self.data_map = dict()

            for col in column_list_to_map:
                self.data_map[col] = {}         

            #Map Non-Numeric Data
            #First convert to pandas dataframe whole dataset

            columns_to_keep = column_list_to_map + numeric_column_list_to_use + labelfields

            #self.data = pd.DataFrame(self.data[1:], columns = self.data[0])
            self.data = self.data.drop([x for x in list(self.data.columns.values) if x not in columns_to_keep], axis=1)

            #Startmapping
            for key in self.data_map.keys():
                counter = 0
                for uniqueitem in [x for x in self.data[key].unique()]:
                    self.data_map[key][uniqueitem] = counter
                    #print(self.data[key])
                    self.data[key] = self.data[key].replace(self.data_map[key])
                    counter += 1
                    
                    
        elif stage == "All_leads" or stage == 'loop':
            if self.leads.shape[0] == 0:
                logging.info("No leads to score, exiting function")
                return
            
            columns_to_keep = ["Id"] + ["Lead_score_timestamp__c"] + column_list_to_map + numeric_column_list_to_use
            
            self.leads = self.leads.drop([x for x in list(self.leads.columns.values) if x not in columns_to_keep], axis=1)
            
            for key in self.data_map.keys():
                logging.debug(key)
                self.ReplaceWithoutMapping = {}
                self.ReplaceWithoutMapping = {x:self.data_map[key]["NO_DATA"] for x in list(self.leads[key].values) if x not in list(self.data_map[key].keys())}
                self.leads[key] = self.leads[key].replace(self.data_map[key])
                if len(self.ReplaceWithoutMapping.keys()) > 0:
                    self.leads[key] = self.leads[key].replace(self.ReplaceWithoutMapping)
    
    
        logging.info("Mapping fields for {} stage".format(stage) + "...Done")
        
    def outlier_specification(self, MapToNumbers, NoMapping):
        
        #To preserve model relatedness to training data, data to be predicted will need to adjere to outliers of training data
        #Outlier specification only needs to be runned once for initial testing, do not run on once model has been built
        
        logging.info("Running outlier specification...")
        
        if hasattr(self, 'outlier_specification'):
            pass
        else:
            self.outlier_specification = None

        self.outlier_specification = {"Device__c":{"min": self.data["Device__c"].min(), "max": self.data["Device__c"].max()},
                                 #"Destination__c":{"min": self.data["Destination__c"].min(), "max": self.data["Destination__c"].max()},
                                 "Country":{"min": self.data["Country"].min(), "max": self.data["Country"].max()},
                                 "IP_city__c":{"min": self.data["IP_city__c"].min(), "max": self.data["IP_city__c"].max()},
                                 "Created_weekday__c":{"min": self.data["Created_weekday__c"].min(), "max": self.data["Created_weekday__c"].max()},
                                 "Last_session_source__c":{"min": self.data["Last_session_source__c"].min(), "max": self.data["Last_session_source__c"].max()},
                                 "LeadSource":{"min": self.data["LeadSource"].min(), "max": self.data["LeadSource"].max()},
                                 "HasOptedOutOfEmail":{"min": self.data["HasOptedOutOfEmail"].min(), "max": self.data["HasOptedOutOfEmail"].max()},
                                 "Have_you_been_to_this_destination_before__c":{"min": self.data["Have_you_been_to_this_destination_before__c"].min(), "max": self.data["Have_you_been_to_this_destination_before__c"].max()},
                                 "Sign_up_source__c":{"min": self.data["Sign_up_source__c"].min(), "max": self.data["Sign_up_source__c"].max()},
                                 "utm_source__c":{"min": self.data["utm_source__c"].min(), "max": self.data["utm_source__c"].max()},
                                 "utm_term__c":{"min": self.data["utm_term__c"].min(), "max": self.data["utm_term__c"].max()},
                                 #"RecordTypeId":{"min": self.data["RecordTypeId"].min(), "max": self.data["RecordTypeId"].max()},
                                }


        
        for x in NoMapping:
            self.outlier_specification[x] = {}    
            print(x)

            #Bellow replacing values for none types to zeros
            #This is appropriate for fields selected, since 0 indicates for filled data set as generally less positive value

            self.data[x] = self.data[x].replace("NO_DATA", 0.0)
            self.leads[x] = self.leads[x].replace("NO_DATA", 0.0)

            self.outlier_specification[x]["min"] = self.data[x].min()
            self.outlier_specification[x]["max"] = self.data[x].mean() + (2 * self.data[x].std()) 
    
        logging.info("Running outlier specification... Done")
    
    def runmodel(self, stage="initial"):
        
        assert stage == "initial" or stage == "All_leads" or stage == "loop"
        
        logging.info("Running model for {} stage...".format(stage))
    
        def standartize_field(datainput, colname, maxvalue, minvalue):
            
            assert datainput[colname].min() >= 0
            
            #Remove outliers
            datainput.loc[datainput[colname] > maxvalue, colname] = maxvalue
            datainput.loc[datainput[colname] < minvalue, colname] = minvalue
            
            #Standartise field
            datainput[colname] = datainput[colname] / maxvalue
            
            
            return datainput
            
        if stage == "initial":
            for col in [x for x in list(self.data.columns.values) if x!='outcome']:
                self.data = standartize_field(self.data, col, self.outlier_specification[col]["max"], self.outlier_specification[col]["min"])
            
            #split data into train and test
            X_train, X_test, y_train, y_test = train_test_split(self.data[self.data.columns.difference(['outcome'])].values, self.data['outcome'].values, test_size=0.33, random_state=42)
            
            #Build Model
            self.model = keras.Sequential([
            #keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax)
            ])

            self.model.compile(optimizer=tf.train.AdamOptimizer(), 
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            self.model.fit(X_train, y_train, epochs=5)
            
            #Test model
            test_loss, test_acc = self.model.evaluate(X_test, y_test)
            
            
            #Save predictions to a file of whole data set
            #prediction = self.model.predict(np.array(self.data[self.data.columns.difference(['outcome'])]))
            
            #prediction = self.model.predict(np.array(self.leads[self.leads.columns.difference(['Id'])]))
            
            #pd.DataFrame(prediction).to_csv("Leads_predicted.csv")
            logging.info('Test accuracy: ' + str(test_acc))
            return print('Test accuracy:', test_acc)
        
            logging.debug("Deleting data used for Learning.")
            delattr(self, 'data')
            
            
            #Return units for training and testing
            #return X_train, X_test, y_train, y_test
            
        elif stage == "All_leads" or stage == "loop":
            if self.leads.shape[0] == 0:
                logging.info("No leads to score, exiting function")
                return
            for col in [x for x in list(self.leads.columns.values) if x!='Id' and x!="id" and x!='Lead_score_timestamp__c']:
                print(col)
                self.leads = standartize_field(self.leads, col, self.outlier_specification[col]["max"], self.outlier_specification[col]["min"])
            
            
            prediction = self.model.predict(np.array(self.leads[self.leads.columns.difference(['Id', 'Lead_score_timestamp__c'])]))
            #pd.DataFrame(prediction).to_csv("Leads_prediction_12.csv")
            
            #Prepare bulk update batches
            #rec_per_batch = 250
            #number_of_batches = math.ceil(self.leads.shape[0] / rec_per_batch)
            
            #open previously written leads
            
            with open(script_path + "InProgressWrittenLeads/" + "written_leads.csv", "r", encoding="UTF-8") as openfile:
                rd = csv.reader(openfile)
                writtenleads = list(rd)
                
                if len(writtenleads) > 1:
                    logging.info("Found previously incomplete lead update, will continue...")
                    writtenleads = [x[0] for x in writtenleads]
                else:
                    logging.info("Starting new lead update from sratch...")
                
            
            for row in range(0, self.leads.shape[0]):
                logging.info("Progress: " + str(round(float(row / self.leads.shape[0]), 2)*100)+"%")
                if np.argmax(prediction[row]) == 0:
                    value_to_push = min(x for x in prediction[row])
                elif np.argmax(prediction[row]) == 1:
                    value_to_push = max(x for x in prediction[row])
                    
                #still need to implement not to update old records
                if self.leads['Id'].iloc[row] not in writtenleads:
                    try:
                        logging.debug(self.sf.Lead.update(self.leads['Id'].iloc[row],{'Lead_score_change__c': str(round(value_to_push, 2))}))
                        logging.debug(self.sf.Lead.update(self.leads['Id'].iloc[row],{'Lead_score_timestamp_2__c': self.leads['Lead_score_timestamp__c'].iloc[row]}))
                        #Append wrote leads to a file for future reference
                        with open(script_path + "InProgressWrittenLeads/" + "written_leads.csv", "a", encoding="UTF-8", newline='') as openfile:
                            wr = csv.writer(openfile)
                            wr.writerows([[self.leads['Id'].iloc[row]]])
                        logging.debug("Updated lead: " + str(self.leads['Id'].iloc[row]) + " with score: " + str(round(value_to_push, 2)))
                        sleep(3)
                    except:
                        logging.debug("Failed updating lead: " + str(self.leads['Id'].iloc[row]) + " with score: " + str(round(value_to_push, 2)))
                        sleep(3)
                else:
                    logging.debug("Skipping already updated: " + str(self.leads['Id'].iloc[row]) + " with score: " + str(round(value_to_push, 2)))
            
            #At this point all leads were attempted to be pushed to salesforce,
            #Therefore writtenleads.csv file needs to be overwritten with empty data, so next time
            #script would not continue, but rather rerun scoring
            logging.info("Scoring leads operation completed, emptying in progress file.")
            with open(script_path + "InProgressWrittenLeads/" + "written_leads.csv", "w", encoding="UTF-8", newline='') as openfile:
                wr = csv.writer(openfile)
                wr.writerows([[""]])
            delattr(self, 'leads')
            
            
        logging.info("Running model for {} stage...".format(stage) + "Done")
        
 #   def UpdateNew(self, Query, sitename, daynumber, exlude_new_days):
 #       if hasattr(self, 'model'):
 #           #Delete old data used for model learning, as it is not needed anymore
 #           
 #       else:
 #           logging.info("Model was not found, please initiate runmodel first to learn from data, before scoring!")
        
        
        
class Looper:
    
    def __init__(self):
        self.HistoricScored = False
    
    def InitiateModel(self):
        logging.info("Looper: Initiating model...")
        self.LeadScoring = LeadScoring(username, password, organizationId)
        self.LeadScoring.QueryData(Query)
        self.LeadScoring.Reframe_data_optional()
        self.LeadScoring.mapfields(MapToNumbers, NoMapping, ['outcome'])
        self.LeadScoring.outlier_specification(MapToNumbers, NoMapping)
        self.LeadScoring.runmodel()
        logging.info("Looper: Initiating model... Done")
        
        
    def ScoreAllHistoricLeadsLeads(self):
        
        #Run bulk update on not converted leads NOTEL still need to add query filter to catch only not populated score fields only
        logging.info("Looper: Scoring historic leads...")
        self.LeadScoring.Reframe_data_optional("All_leads")
        self.LeadScoring.mapfields(MapToNumbers, NoMapping, ['outcome'], "All_leads")
        self.LeadScoring.runmodel("All_leads")
        logging.info("Looper: Scoring historic leads... Done")
        self.HistoricScored == True
        
    def StartLooper(self):
        while True:
            if hasattr(self, 'LeadScoring'):
                delattr(self, 'LeadScoring')
                logging.info("Restarting looper and Reevaluating the model...")
                
            self.InitiateModel()
            
            if self.HistoricScored == False:
                self.ScoreAllHistoricLeadsLeads()
            
            while self.LeadScoring.ObjectCreationDateTimestamp > (datetime.now() - timedelta(days=reevaluate_model_after_days)):
                #module to score new leads
                logging.info("Sleeping (seconds): " + str(loop_sleep_seconds))
                sleep(loop_sleep_seconds)
                logging.info("Looper: Scoring changed leads...")
                self.LeadScoring.QueryData(Query_loop, Query_opp = None, stage="loop", sitename=sitename)
                self.LeadScoring.Reframe_data_optional('loop')
                self.LeadScoring.mapfields(MapToNumbers, NoMapping, ['outcome'], "loop")
                self.LeadScoring.runmodel("loop")
                logging.info("Looper: Scoring changed leads...")
        

if __name__ == "__main__":
    #Run Program
    job = Looper().StartLooper()
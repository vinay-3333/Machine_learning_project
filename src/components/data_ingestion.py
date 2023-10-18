import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


#Intitialize the data ingetion configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifcats','train.csv')
    test_data_path:str=os.path.join('artifcats','test.csv')
    raw_data_path:str=os.path.join('artifcats','raw.csv')

#create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df = pd.read_csv('notebook/data/gemstone.csv')
            logging.info('Data read as pandas data frame')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('train test split')
            train_set,test_set = train_test_split(df,test_size=0.30)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            #data ingestion is complete
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info('Exception occured at data ingestion')
            raise CustomException(e,sys)



import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifcats','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variable from train and test data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet()
            }
            model_report : dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n',"="*35)
            logging.info(f"model Report : {model_report} ")

            # To Get best model score form dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best model Found , Model Name : {best_model_name}, R2 score : {best_model_score}")
            print('\n','='*35)
            logging.info(f"Best model Found , Model Name : {best_model_name}, R2 score : {best_model_score}")


            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )


        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)
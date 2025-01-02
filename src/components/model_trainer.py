import sys
import os
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split Train and test data")
            
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info(x_train)
            models={
                "RandomForest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "K-neighbours":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBossting":CatBoostRegressor(),
                "AdaBoost":AdaBoostRegressor()
            }
            model_report:dict=evaluate_model(X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            best_model_names=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_names]
            if best_model_score<0.6:
                raise CustomException("No best Model Found")
            logging.info("Best Model Found by training the data.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            predicted_val=r2_score(y_test,predicted)
            return predicted_val
        
        except Exception as e:
            raise CustomException(e,sys)
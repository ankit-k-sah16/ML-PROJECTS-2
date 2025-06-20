import os
import sys

from dataclasses import dataclass
from catboost  import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from source.logger import logging
from source.exception import customException
from source.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Neighbours": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "AdaBoost":AdaBoostRegressor(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,
                                              y_test=y_test,models=models)
            
            # to get the best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # to get the best model name from the dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise customException("No best model found")
            logging.info(f"Best found Model  on both training data and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            R2_score=r2_score(y_test,predicted)
            return R2_score
        
        except Exception as e:
            raise customException(e,sys)
        


















import os
import sys
from dataclasses import dataclass, field

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

BEST_MODEL_SELECTION_THRESHOLD = 0.6

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer():
    """Trains a model and saves it to a file.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostClassifier(),
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get the best model score from the model report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the model report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<BEST_MODEL_SELECTION_THRESHOLD:
                raise CustomException("No best model found for the given threshold")

            logging.info(f"Best model found on train and test dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"Model saved to disk successfully")

            score = r2_score(y_test, best_model.predict(X_test))
            return score

        except Exception as e:
            raise CustomException(e, sys)
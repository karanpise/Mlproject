import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            logging.info("Splitting training and testing data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(random_state=42, verbosity=0),
                "CatBoost": CatBoostRegressor(verbose=False, random_state=42)
            }

            logging.info("Evaluating Models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            print(model_report)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best Model: {best_model_name}, R2 Score: {best_model_score}"
            )

            # Fit the best model before saving
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_trasformation import DataTransformation

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    train_arr, test_arr = transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    trainer = ModelTrainer()

    score = trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("R2 Score:", score)
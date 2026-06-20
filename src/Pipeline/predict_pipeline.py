import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class CustomData:
    age: int
    annual_income: float
    spending_score: int

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Annual_Income": [self.annual_income],
                "Spending_Score": [self.spending_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    _model = None
    _preprocessor = None

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            if PredictPipeline._model is None:
                PredictPipeline._model = load_object(file_path=self.model_path)

            if PredictPipeline._preprocessor is None:
                PredictPipeline._preprocessor = load_object(
                    file_path=self.preprocessor_path
                )

            transformed_features = PredictPipeline._preprocessor.transform(features)
            prediction = PredictPipeline._model.predict(transformed_features)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

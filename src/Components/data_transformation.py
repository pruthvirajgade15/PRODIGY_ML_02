import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Age", "Annual_Income", "Spending_Score"]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info("Numerical columns standard scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            drop_columns = ["CustomerID"]
            input_feature_train_df = train_df.drop(columns=drop_columns)
            input_feature_test_df = test_df.drop(columns=drop_columns)

            feature_columns = ["Age", "Annual_Income", "Spending_Score"]
            input_feature_train_df = input_feature_train_df[feature_columns]
            input_feature_test_df = input_feature_test_df[feature_columns]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.array(input_feature_train_arr)
            test_arr = np.array(input_feature_test_arr)

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

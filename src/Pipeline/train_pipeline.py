import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            train_data_path, test_data_path, raw_data_path = (
                self.data_ingestion.initate_data_ingestion()
            )
            logging.info(f"Raw data saved at: {raw_data_path}")

            train_array, test_array, preprocessor_path = (
                self.data_transformation.initiate_data_transformation(
                    train_data_path,
                    test_data_path,
                )
            )
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            silhouette_score = self.model_trainer.initiate_model_trainer(
                train_array,
                test_array,
            )

            logging.info(
                f"Training pipeline completed with silhouette score: {silhouette_score}"
            )
            return silhouette_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"Training completed. Silhouette score: {score}")

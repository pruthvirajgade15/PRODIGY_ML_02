import os
import sys
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Training K-Means model on scaled training data")

            best_k = 5
            best_score = -1

            for k in range(4, 8):
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(train_array)
                score = silhouette_score(train_array, labels)

                logging.info(f"K-Means k={k} silhouette score: {score}")

                if score > best_score:
                    best_score = score
                    best_k = k

            best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            best_model.fit(train_array)

            test_labels = best_model.predict(test_array)
            test_score = silhouette_score(test_array, test_labels)

            logging.info(f"Best K-Means model found with k={best_k}")
            logging.info(f"Train silhouette score: {best_score}")
            logging.info(f"Test silhouette score: {test_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return test_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    from src.Components.data_transformation import DataTransformation

    data_ingestion = DataIngestion()
    train_data_path, test_data_path, _ = data_ingestion.initate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path,
        test_data_path,
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))

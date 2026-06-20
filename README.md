# PRODIGY_ML_02 - Mall Customer Segmentation

Customer segmentation project using K-Means clustering on the Mall Customers dataset. The project includes data ingestion, preprocessing, model training, saved artifacts, a prediction pipeline, and a Streamlit app.

## Project Structure

```text
PRODIGY_ML_02/
│
├── app.py
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── logs/
│   └── log files generated while running components
│
└── src/
    ├── __init__.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    │
    ├── Components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    │
    ├── Pipeline/
    │   ├── __init__.py
    │   ├── train_pipeline.py
    │   └── predict_pipeline.py
    │
    └── Notebooks/
        ├── 1.EDA.ipynb
        ├── 2.Model_Training.ipynb
        └── Data/
            ├── Mall_Customers.csv
            └── Mall_Customers_cleaned.csv
```

## Setup

```powershell
pip install -r requirements.txt
```

## Train the Model

```powershell
python src\Pipeline\train_pipeline.py
```

This creates:

```text
artifacts/data.csv
artifacts/train.csv
artifacts/test.csv
artifacts/preprocessor.pkl
artifacts/model.pkl
```

## Run Streamlit App

```powershell
streamlit run app.py
```

Or:

```powershell
python -m streamlit run app.py
```

## Input Features

| Feature | Description |
| --- | --- |
| `Age` | Customer age |
| `Annual_Income` | Annual income in thousands |
| `Spending_Score` | Spending score from 1 to 100 |

## Author

Pruthviraj

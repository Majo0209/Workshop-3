
---

# Workshop 3 – ETL Process using Apache Kafka and Machine Learning

## Project Overview

This project builds a complete ETL–ML–Streaming–Dashboard pipeline to predict the Happiness Score of different countries from 2015 to 2019.
The workflow includes data processing, model training, real-time streaming with Apache Kafka, and an interactive dashboard made with Streamlit.

The goal is to train a regression model that predicts happiness levels using socioeconomic variables, and then deploy it in a streaming environment connected to a MySQL database.

---

## System Objective

Develop a predictive system that integrates the following steps:

* Data extraction and transformation from multiple datasets.
* Model training and evaluation with regression models.
* Real-time data streaming using Apache Kafka.
* Data storage in a MySQL database.
* Visualization of model performance using Streamlit.

---

## Project Structure

```
WORKSHOP 3/
│
├── data/
│   ├── processed/
│   │   ├── happiness_model.csv
│   │   ├── happiness_predictions_Ridge.csv
│   │   ├── 2015.csv
│   │   ├── 2016.csv
│   │   ├── 2017.csv
│   │   ├── 2018.csv
│   │   └── 2019.csv
│
├── models/
│   └── Best_Ridge_Model.pkl
│
├── consumer.py
├── producer.py
├── dashboard_model.py
├── Eda.ipynb
├── Regression_Model.ipynb
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python 3.12
* Apache Kafka
* MySQL Workbench
* Streamlit
* Main Python libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, statsmodels, joblib, kafka-python, sqlalchemy, pymysql

---

## How to Run the Project

### 0. Clone the repository

```bash
git clone https://github.com/Majo0209/Workshop-3.git
cd Workshop-3
```

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/Scripts/activate      # On Windows
pip install -r requirements.txt
```

### 2. Run the notebooks for data analysis and model training

1. Open and execute:

   * `Eda.ipynb` → Data cleaning and dataset merge.
   * `Regression_Model.ipynb` → Model training and evaluation.
2. The following files will be created:

   * `data/processed/happiness_model.csv`
   * `models/Best_Ridge_Model.pkl`
   * `data/processed/happiness_predictions_Ridge.csv`

### 3. Start Apache Kafka

In separate terminals:

```bash
zookeeper-server-start.bat config/zookeeper.properties
kafka-server-start.bat config/server.properties
```

Create the topic:

```bash
kafka-topics.bat --create --topic happiness_topic --bootstrap-server localhost:9092
```

### 4. Run Producer and Consumer scripts

```bash
python producer.py
python consumer.py
```

The Producer sends the transformed data to the topic, and the Consumer receives it and saves the predictions in MySQL (`happiness_dw.predicciones_happiness`).

### 5. Run the Streamlit dashboard

```bash
streamlit run dashboard_model.py
```

This will open the Streamlit dashboard showing model metrics, graphs, and comparisons.

---

## Main Results

### Model Metrics (Ridge Regression)

| Metric | Value  |
| ------ | ------ |
| R²     | 0.770  |
| RMSE   | 0.544  |
| MAE    | 0.429  |
| MAPE   | 8.51 % |
| Bias   | 0.011  |

### Streamlit Visualizations

* Real vs Predicted Score: shows strong alignment between actual and predicted values.
* Annual Average of Scores: shows how the model follows yearly trends.
* Error Distribution: centered around zero, no bias.
* Feature Importance: the most important variables are Freedom, Health Life Expectancy, and GDP per Capita.

---

## System Workflow

1. EDA and transformation: unify and clean yearly datasets (2015–2019).
2. Model training: compare models and select Ridge Regression.
3. Streaming: Producer → Kafka → Consumer → MySQL.
4. Visualization: analyze metrics and results with the Streamlit dashboard.


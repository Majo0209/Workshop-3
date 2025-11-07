
---

# Workshop 3 – ETL Process using Apache Kafka and Machine Learning

## 1. Project Overview

This project builds a complete **ETL–ML–Streaming–Dashboard pipeline** to predict the *Happiness Score* of different countries from 2015 to 2019.
The workflow includes **data processing, model training, real-time streaming with Apache Kafka, and an interactive dashboard created with Streamlit**.

The main goal is to train a regression model that predicts happiness levels using socioeconomic variables, and deploy it in a streaming environment connected to a MySQL database.

---

## 2. System Objectives

Develop a predictive system that integrates the following steps:

* Data extraction and transformation from multiple datasets.
* Model training and evaluation using regression models.
* Real-time data streaming through Apache Kafka.
* Data storage and management in MySQL.
* Visualization of model performance using Streamlit.

---

## 3. Block Diagram

<img width="934" height="302" alt="image" src="https://github.com/user-attachments/assets/a13cfc48-de5a-42a4-a0ef-6b8b5b60b455" />

**Pipeline Summary:**

1. **EDA and Transformation:** unify and clean datasets (2015–2019).
2. **Model Training:** compare models and select Ridge Regression.
3. **Streaming:** Producer → Kafka → Consumer → MySQL.
4. **Visualization:** analyze model results through the Streamlit dashboard.

---

## 4. Project Structure

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

## 5. Technologies Used

* **Python 3.12**
* **Apache Kafka** – real-time streaming
* **MySQL Workbench** – database management
* **Streamlit** – interactive dashboard visualization
* **Main Python libraries:**
  `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `statsmodels`, `joblib`, `kafka-python`, `sqlalchemy`, `pymysql`

---

## 6. How to Install Apache Kafka on Windows

Kafka may have issues running directly on Windows because it was designed for Linux.
If you are using Windows, follow these steps to install and run Kafka properly:

### Option 1 – For Windows 10 or newer

Use **WSL (Windows Subsystem for Linux)** to simulate a Linux environment.

### Option 2 – For older Windows versions

Use **Docker** to run Kafka inside a container.

> It is not recommended to run Kafka only with the JVM on Windows because some required Linux features are missing.

---

### Step 1: Install WSL

Check your Windows version: press `Win + R`, type `winver`, and click OK.
Make sure it is **Windows 10 version 2004 (Build 19041) or higher**.
Then open **Command Prompt (Administrator)** and run:

```bash
wsl --install
```

---

### Step 2: Install Java

Check if Java is installed:

```bash
java -version
```

If not, download it from:
[https://www.oracle.com/java/technologies/downloads/](https://www.oracle.com/java/technologies/downloads/)

---

### Step 3: Download Apache Kafka

Go to the official site:
[https://kafka.apache.org/downloads](https://kafka.apache.org/downloads)

Unzip the files in a folder (for example, `C:\kafka`) and add the environment variable:

```
KAFKA_HOME = C:\kafka
```

---

### Step 4: Start Zookeeper

Kafka uses Zookeeper to manage brokers. Open **Command Prompt** and run:

```bash
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

Check if it’s running:

```bash
netstat -an | findstr 2181
```

If port `2181` is listening, Zookeeper is working.

---

### Step 5: Start Kafka Server

In another terminal, run:

```bash
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

Then check:

```bash
netstat -an | findstr 9092
```

If port `9092` is listening, Kafka is active.

---

## 7. How to Run the Project

### Step 0: Clone the repository

```bash
git clone https://github.com/Majo0209/Workshop-3.git
cd Workshop-3
```

### Step 1: Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/Scripts/activate      # On Windows
pip install -r requirements.txt
```

### Step 2: Run notebooks for EDA and model training

1. Execute:

   * `Eda.ipynb` → Data cleaning and dataset merge.
   * `Regression_Model.ipynb` → Train and evaluate the Ridge model.
2. Generated files:

   * `data/processed/happiness_model.csv`
   * `models/Best_Ridge_Model.pkl`
   * `data/processed/happiness_predictions_Ridge.csv`

---

### Step 3: Start Kafka services

In two terminals:

```bash
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

Create the topic:

```bash
kafka-topics.bat --create --topic happiness_topic --bootstrap-server localhost:9092
```

---

### Step 4: Run Producer and Consumer scripts

```bash
python producer.py
python consumer.py
```

The **Producer** sends the transformed data to the topic, and the **Consumer** receives it and saves predictions into MySQL (`happiness_dw.predicciones_happiness`).

---

### Step 5: Run the Streamlit Dashboard

```bash
streamlit run dashboard_model.py
```
Once running, open the dashboard at:

* Local URL: http://localhost:8501
* Network URL: http://192.168.80.11:8501

This opens the interactive Streamlit dashboard with model metrics, charts, and comparisons.


---

## 8. Main Results

### Ridge Regression Model Metrics

| Metric | Value  |
| ------ | ------ |
| R²     | 0.770  |
| RMSE   | 0.544  |
| MAE    | 0.429  |
| MAPE   | 8.51 % |
| Bias   | 0.011  |

### Streamlit Dashboard Visualizations

* **Real vs Predicted Score**: alignment between actual and predicted values.

* **Annual Average of Scores**: consistent performance across years.

* **Average Score per Year**: predicted means closely match real annual values.

* **Error Distribution**: centered around zero, indicating no bias.

* **Feature Importance**: top factors are Freedom, Health Life Expectancy, and GDP per Capita.

* **Global Map**: displays predicted happiness by country for geographic comparison.


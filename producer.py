import pandas as pd
import numpy as np
import joblib
from kafka import KafkaProducer
import json
import time
import glob
import os
from sklearn.model_selection import train_test_split

model_files = glob.glob("models/Best_*_Model.pkl")
if not model_files:
    raise FileNotFoundError("No se encontró ningún modelo en 'models/'. Ejecuta primero el notebook de regresión.")

model_path = max(model_files, key=os.path.getmtime)
model_name = os.path.basename(model_path).replace("Best_", "").replace("_Model.pkl", "")
print(f"Modelo detectado automáticamente: {model_name}")

best_model = joblib.load(model_path)
print("Modelo cargado correctamente.")


csv_base = "data/processed/happiness_model.csv"
if not os.path.exists(csv_base):
    raise FileNotFoundError(f"No se encontró el dataset base: {csv_base}")

df = pd.read_csv(csv_base)
print(f"Dataset base cargado correctamente: {len(df)} filas, {len(df.columns)} columnas")

X = df[[
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "perceptions_of_corruption",
    "year"
]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

df_all = df.copy()
df_all["score_predicho"] = np.nan
df_all["is_train"] = 0
df_all["is_test"] = 0

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

df_all["score_real"] = y.values
df_all.loc[X_train.index, "score_predicho"] = y_pred_train
df_all.loc[X_test.index, "score_predicho"] = y_pred_test

df_all.loc[X_train.index, "is_train"] = 1
df_all.loc[X_test.index, "is_test"] = 1

cols = [
    "country", "gdp_per_capita", "social_support",
    "health_life_expectancy", "freedom", "perceptions_of_corruption",
    "year", "score_real", "score_predicho", "is_train", "is_test"
]
df_all = df_all[cols]

csv_path = f"data/processed/happiness_predictions_{model_name}.csv"
df_all.to_csv(csv_path, index=False)
print(f"Archivo con predicciones guardado: {csv_path}")
print(f"Filas totales: {len(df_all)}")

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks='all',
    linger_ms=5
)

TOPIC = "happiness_topic"
count = 0

for i, row in df_all.iterrows():
    message = {col: (float(row[col]) if isinstance(row[col], (int, float)) and not pd.isna(row[col]) else str(row[col]))
               for col in cols}
    future = producer.send(TOPIC, value=message)
    future.get(timeout=60)
    count += 1

    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(df_all)} registros enviados...")

producer.flush()
producer.close()
print(f"Total de registros enviados a Kafka: {count} / {len(df_all)}")

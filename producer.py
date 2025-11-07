import pandas as pd
import joblib
import statsmodels.api as sm
from kafka import KafkaProducer
import json
import time
import glob
import os

# ---------------------------------------------------------------------
# 1. Detectar el modelo más reciente (.pkl)
# ---------------------------------------------------------------------
model_files = glob.glob("models/Best_*_Model.pkl")
if not model_files:
    raise FileNotFoundError("No se encontró ningún modelo en 'models/'. Ejecuta primero el notebook de regresión.")

model_path = max(model_files, key=os.path.getmtime)
model_name = os.path.basename(model_path).replace("Best_", "").replace("_Model.pkl", "")
print(f"Modelo detectado automáticamente: {model_name}")

model = joblib.load(model_path)
print("Modelo cargado correctamente.")

# ---------------------------------------------------------------------
# 2. Cargar el CSV de predicciones correspondiente
# ---------------------------------------------------------------------
csv_path = f"data/processed/happiness_predictions_{model_name}.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Datos cargados correctamente: {len(df)} filas")

# ---------------------------------------------------------------------
# 3. Configurar Kafka producer
# ---------------------------------------------------------------------
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks='all',  # espera confirmación del broker
    linger_ms=5,  # pequeño delay para agrupar mensajes
)

TOPIC = "happiness_topic"

# ---------------------------------------------------------------------
# 4. Enviar todos los registros con confirmación
# ---------------------------------------------------------------------
cols = [
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "perceptions_of_corruption",
    "year",
    "score_real",
    "score_predicho",
    "is_train",
    "is_test"
]

count = 0
for i, row in df.iterrows():
    message = {col: (float(row[col]) if isinstance(row[col], (int, float)) else row[col]) for col in cols}
    
    # enviar y esperar confirmación
    future = producer.send(TOPIC, value=message)
    future.get(timeout=60)  # bloquea hasta que Kafka confirme el envío
    
    count += 1
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(df)} registros enviados...")

# ---------------------------------------------------------------------
# 5. Asegurar que todo fue entregado
# ---------------------------------------------------------------------
producer.flush()
producer.close()
print(f"Total de registros enviados a Kafka: {count} / {len(df)}")

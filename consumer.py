import json
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy import create_engine
import time

KAFKA_TOPIC = "happiness_topic"
KAFKA_SERVER = "localhost:9092"

MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
DW_SCHEMA = "happiness_dw"


engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DW_SCHEMA}"
)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_SERVER],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id="happiness_group",
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print(f"Connected to Kafka topic '{KAFKA_TOPIC}'.")
print("Waiting for messages from the producer...\n")

batch = []
batch_size = 50
table = "predicciones_happiness"
total_expected = 771  
received = 0

try:
    for msg in consumer:
        batch.append(pd.DataFrame([msg.value]))
        received += 1

        if len(batch) >= batch_size:
            df_batch = pd.concat(batch, ignore_index=True)
            with engine.begin() as conn:
                df_batch.to_sql(table, conn, if_exists="append", index=False)
            print(f"{len(df_batch)} records inserted into '{table}' (total: {received}/{total_expected}).")
            batch = []

        if received >= total_expected:
            print(f"\nAll {total_expected} records have been received. Stopping consumer.")
            break

except KeyboardInterrupt:
    print("Consumption manually stopped by user.")

finally:
    if batch:
        df_batch = pd.concat(batch, ignore_index=True)
        with engine.begin() as conn:
            df_batch.to_sql(table, conn, if_exists="append", index=False)
        print(f"{len(df_batch)} final records inserted into '{table}'.")

    consumer.close()
    print("Consumer closed successfully.")
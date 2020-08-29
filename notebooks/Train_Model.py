# Databricks notebook source
# MAGIC %md ## Load Historical Data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data")

# COMMAND ----------

from random import random, randint
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
    params = {"n_estimators": 5, "random_state": 42}
    sk_learn_rfr = RandomForestRegressor(**params)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_param("param_1", randint(0, 100))
    mlflow.log_metrics({"metric_1": random(), "metric_2": random() + 1})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=sk_learn_rfr,
        artifact_path="sklearn-model",
        registered_model_name="sk-learn-random-forest-reg-model"
    )

# COMMAND ----------

result = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    "sk-learn-random-forest-reg"
)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.create_registered_model("sk-learn-random-forest-reg-model")

# COMMAND ----------

client = MlflowClient()
result = client.create_model_version(
    name="sk-learn-random-forest-reg-model",
    source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    run_id="d16076a3ec534311817565e6527539c0"
)

# COMMAND ----------



# COMMAND ----------


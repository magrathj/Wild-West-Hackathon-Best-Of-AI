# Databricks notebook source
# MAGIC %md
# MAGIC # Apply fraud algorithm to streaming data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Replace with stream
# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data")

# COMMAND ----------

# DBTITLE 1,Load production model from MLflow Registry 
import mlflow.pyfunc

model_name = "sk-learn-random-forest-reg-model"
stage = 'Production'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}
)

# COMMAND ----------

# DBTITLE 1,Apply prediction
predicted_df = model.predict(df)

# COMMAND ----------

# DBTITLE 1,Filter out those which are considered fraudualent and write them to a table
predicted_df.filter('prediction==1').write.format("delta")
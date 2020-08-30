# Databricks notebook source
# MAGIC %md
# MAGIC # Apply fraud algorithm to streaming data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Replace with stream
# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data").withColumn('Id', monotonically_increasing_id()).withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))

# COMMAND ----------

# DBTITLE 1,Load production model from MLflow Registry 
import mlflow.pyfunc

model_name = "RulesBasedModel"
stage = 'Production'

model_version_uri = "models:/{model_name}/{stage}".format(model_name=model_name, stage=stage) 

# COMMAND ----------

sc.setCheckpointDir('/tmp/checkpoint')

# COMMAND ----------

output_df = df.checkpoint()

# COMMAND ----------

outSchema = output_df.schema

# COMMAND ----------

outSchema = outSchema.add('prediction', StringType())

# COMMAND ----------

# DBTITLE 1,Create Pandas UDF to call model
import pandas as pd

def predict(iterator):
    # pdf is a pandas.DataFrame
    model = mlflow.pyfunc.load_model(
        model_uri=model_version_uri
    )    
    for features in iterator:
      yield pd.DataFrame(model.predict(features))

# COMMAND ----------

# DBTITLE 1,Apply prediction
predicted_df = df.mapInPandas(predict, outSchema)

# COMMAND ----------

# DBTITLE 1,Filter out those which are considered fraudualent and write them to a table
predicted_df.filter('prediction=="True"').write.saveAsTable(model_name+'_PossibleFraud')

# COMMAND ----------

# DBTITLE 1,Save dataframe which has had Rules based model applied to it
predicted_df.write.saveAsTable(model_name)

# COMMAND ----------

display(spark.read.table('PossibleFraud'))
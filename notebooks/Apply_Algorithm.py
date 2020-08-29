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

outSchema = df.schema

# COMMAND ----------

outSchema = outSchema.add('prediction', StringType())

# COMMAND ----------

len(outSchema)

# COMMAND ----------

outSchema

# COMMAND ----------

# DBTITLE 1,Create Pandas UDF to call model
def predict(iterator):
    # pdf is a pandas.DataFrame
    model = mlflow.pyfunc.load_model(
        model_uri=model_version_uri
    )    
    for features in iterator:
      yield pd.DataFrame(model.predict(features))

# COMMAND ----------

df.mapInPandas(predict, "test string").show(1)

# COMMAND ----------

mlflow.pyfunc.load_pyfunc(model_version_uri)

# COMMAND ----------

# DBTITLE 1,Apply prediction
display(df.withColumn('prediction', test('PriceDifference')))

# COMMAND ----------

# DBTITLE 1,Filter out those which are considered fraudualent and write them to a table
predicted_df.filter('prediction==1').write.format("delta")
# Databricks notebook source
# MAGIC %md
# MAGIC # Apply fraud algorithm to streaming data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow

# COMMAND ----------

# DBTITLE 1,Replace with stream
# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data").withColumn('Id', monotonically_increasing_id()).withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))

# COMMAND ----------

# DBTITLE 1,Cache Dataframe
df.cache()
df.count()

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build a Rules Based Model

# COMMAND ----------

# DBTITLE 1,Create Pyfunc Class for RulesBasedModel
from mlflow.pyfunc import PythonModel
import pandas as pd

class RulesBasedModel(PythonModel):
    import pandas as pd

    def __init__(self, PriceDifferenceThreshold, ItemCountThreshold):
        self.PriceDifferenceThreshold = PriceDifferenceThreshold
        self.ItemCountThreshold = ItemCountThreshold

    def predict(self, context, data):
        data['prediction'] = pd.DataFrame(data['PriceDifference'].apply(lambda x: 'True' if x > self.PriceDifferenceThreshold else 'False'))
        return data

# COMMAND ----------

conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'mlflow=1.10.0',
                'numpy=1.16.5',
                'python=3.6.9',
                'pandas'
            ],
            'name': 'mlflow-env'
        }

# COMMAND ----------

mlflow.set_experiment("/Users/wildwesthacker42@bpcs.com/RulesBasedModel")

# COMMAND ----------

# DBTITLE 1,Start an experiment and log the model
import mlflow.pyfunc

with mlflow.start_run(run_name="RulesBasedModel") as run:
  
  pyfunc_model = RulesBasedModel(PriceDifferenceThreshold=200, ItemCountThreshold=2)
  
  artifact_path = "model"
  mlflow.pyfunc.log_model(python_model=pyfunc_model, artifact_path=artifact_path, conda_env=conda_env)

  run_id = mlflow.active_run().info.run_id

  model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

# COMMAND ----------

# DBTITLE 1,Assert model predicts something
model = mlflow.pyfunc.load_model(
        model_uri=model_uri
    )   
assert len(model.predict(df.toPandas())) > 0 

# COMMAND ----------

# DBTITLE 1,Register the model
result=mlflow.register_model(model_uri, "RulesBasedModel")

# COMMAND ----------

# DBTITLE 1,Wait until model is registered
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)

wait_until_ready(result.name, result.version)

# COMMAND ----------

print(f"Model version: {result.version}, Model name:{result.name}")

# COMMAND ----------

# DBTITLE 1,Promote model to Production stage
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(result.name, result.version, 'Staging')

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Rules Based Model on the data set to get labelled data to train with

# COMMAND ----------

# DBTITLE 1,Load new Staging model from MLflow Registry 
import mlflow.pyfunc

model_name = "RulesBasedModel"
stage = 'Staging'

model_version_uri = "models:/{model_name}/{stage}".format(model_name=model_name, stage=stage) 

# COMMAND ----------

# DBTITLE 1,Create checkpoint to prevent lazy evaluation adding changes to df schema
sc.setCheckpointDir('/tmp/checkpoint')
output_df = df.checkpoint()

# COMMAND ----------

# DBTITLE 1,Add new column prediction
outSchema = output_df.schema
outSchema = outSchema.add('prediction', StringType())

# COMMAND ----------

len(outSchema)

# COMMAND ----------

# DBTITLE 1,Create Pandas UDF to call model
def predict(iterator):
    import pandas as pd
    # pdf is a pandas.DataFrame
    model = mlflow.pyfunc.load_model(
        model_uri=model_version_uri
    )    
    for features in iterator:
      yield pd.DataFrame(model.predict(features))

# COMMAND ----------

# DBTITLE 1,Apply prediction
predicted_df = df.mapInPandas(predict, output_df.schema)

# COMMAND ----------

# DBTITLE 1,Assert Pandas UDF predicts something
assert predicted_df.count() > 0

# COMMAND ----------

# DBTITLE 1,Filter out those which are considered fraudualent and write them to a table
predicted_df.filter('prediction=="True"').write.mode("overwrite").saveAsTable(model_name+'_PossibleFrauds')

# COMMAND ----------

# DBTITLE 1,Save dataframe which has had Rules based model applied to it
predicted_df.write.mode("overwrite").saveAsTable(model_name)

# COMMAND ----------

display(spark.read.table(model_name+'_PossibleFrauds'))

# COMMAND ----------

display(spark.read.table(model_name))
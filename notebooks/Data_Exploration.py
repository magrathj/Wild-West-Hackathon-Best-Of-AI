# Databricks notebook source
# MAGIC %md # Welcome to Quest #3 of Challenge 2.  
# MAGIC 
# MAGIC This notebook will comprise all of Quest #3. We suggest that your steps should be data exploration, planning, write algorithm, apply algorithm, and finally business explanation. If you are not completely confident after researching, we also suggest that once you have a plan of attack to complete the quest that you send us you plan. We can help fill in the gaps.
# MAGIC 
# MAGIC The Adventure Works executive and finance teams have had some unexpected financial results. They suspect foul play! You have been hired to find examples of fraud and report when it happens in real-time. 
# MAGIC 
# MAGIC 1) Adventure works only receives a total charge for each customer order from each rep. This is because the reps have the freedom to give discretionary discounts to customers if they think it will sway them to but product. The executive team thinks that there is rampant fraud amoung the reps. They suspect that the reps are adding items to a customers order and then taking that item home with them. Adventure works needs you to establish a pattern in finding this behavior then compile a list, moving forward, of fraud examples. 
# MAGIC 
# MAGIC 2) Adventure Works let go of their systems administrator a few years back. The systems admin was very unhappy with this and the executive team heard from a co-worker that the sys admin is a known hacker. The executive team is scared that the sys admin left something behind in the system. If you can find the discrepancy in real time then it will be easier for engineers to track down the hack.    
# MAGIC 
# MAGIC **Note** there exists only two types of fraud present in the data following the scenario above should make them very evident.
# MAGIC 
# MAGIC This quest entails identifying the fraud trends and writing an algorithm to find fraud in real time. 
# MAGIC 
# MAGIC You will be judged by accuracy in finding real-time fraud, as well as your business explanation.

# COMMAND ----------

# MAGIC %md ## Load Historical Data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data")
df = df.withColumn('Id', monotonically_increasing_id()).withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))

# COMMAND ----------

df.cache()
df.count()

# COMMAND ----------

# MAGIC %md ## Data Exploration
# MAGIC Find the fraud trends. Spend most of your time here!

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

display(df.groupBy('date').agg({"ListPrice": "sum", "StandardCost": "sum", "ChargedPrice": "sum", "ItemCount": "sum"}).orderBy('date'))

# COMMAND ----------

display(df.withColumn('year', year('date')).withColumn('month', month('date')).groupBy('year', 'month').agg({"ListPrice": "sum", "StandardCost": "sum", "ChargedPrice": "sum", "ItemCount": "sum"}).orderBy('year', 'month'))

# COMMAND ----------

display(df.groupBy('RepName').agg({"ListPrice": "mean", "StandardCost": "mean", "ChargedPrice": "mean", "ItemCount": "mean"}))

# COMMAND ----------

display(df.groupBy('RepName').agg({"ListPrice": "sum", "StandardCost": "sum", "ChargedPrice": "sum", "ItemCount": "sum"}))

# COMMAND ----------

display(df.groupBy('RepName').count().orderBy(asc('count')))

# COMMAND ----------

display(df.groupBy('RepName').avg('ChargedPrice'))

# COMMAND ----------

display(df.groupBy('RepName').max('ChargedPrice'))

# COMMAND ----------

display(df.groupBy('RepName').min('ChargedPrice'))

# COMMAND ----------

display(df.filter(col('ChargedPrice') > 13000).groupBy('RepName').count())

# COMMAND ----------

display(df.filter(col('ChargedPrice') > 13000).withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice') ).select('RepName', 'ListPrice', 'PriceDifference', 'ChargedPrice', 'StandardCost'))

# COMMAND ----------

display(df.groupBy('RepName', 'CustId').agg({"ItemCount": "max"}).filter('max(ItemCount) > 2'))

# COMMAND ----------

display(df
        .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
        .groupBy('RepName', 'CustId')
        .agg({"ItemCount": "max"})
        .withColumnRenamed('max(ItemCount)','MaxItemCount')
        .orderBy(desc('MaxItemCount'))
       )

# COMMAND ----------

grouped_df = (df
              .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
              .groupBy('RepName', 'CustId')
              .agg({"ItemCount": "max"})
              .withColumnRenamed('max(ItemCount)','MaxItemCount')
              .orderBy(desc('MaxItemCount'))
             )

# COMMAND ----------

display(df
        .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
        .join(grouped_df, [df.RepName==grouped_df.RepName, df.CustId==grouped_df.CustId])
        .select(df.RepName, df.CustId, 'ListPrice', 'PriceDifference', 'ChargedPrice', 'StandardCost', 'ItemCount')
        .selectExpr('*', 'ListPrice/ItemCount as AvgListPrice', 'ChargedPrice/ItemCount as AvgChargedPrice')
        .selectExpr('*', 'AvgListPrice - AvgChargedPrice as AvgDiff')
        .filter('AvgDiff > 10')
       )

# COMMAND ----------

display(df
        .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
        .join(grouped_df, [df.RepName==grouped_df.RepName, df.CustId==grouped_df.CustId])
        .select(df.RepName, df.CustId, 'ListPrice', 'PriceDifference', 'ChargedPrice', 'StandardCost', 'ItemCount')
        .selectExpr('*', 'ListPrice/ItemCount as AvgListPrice', 'ChargedPrice/ItemCount as AvgChargedPrice')
        .selectExpr('*', 'AvgListPrice - AvgChargedPrice as AvgDiff')
        .filter('AvgDiff > 100')
        .groupBy('RepName')
        .count()
       )

# COMMAND ----------

display(df.withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice')).select('RepName', 'ListPrice', 'PriceDifference', 'ChargedPrice', 'StandardCost'))

# COMMAND ----------

display(df
        .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
        .select('ItemCount', 'RepName', 'ListPrice', 'PriceDifference', 'ChargedPrice', 'StandardCost'))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Planning
# MAGIC Plan how your algorithm will find the fraud trends in real-time.

# COMMAND ----------

# MAGIC %md
# MAGIC * Analyse what potential columns would identicate fraud - as we dont have defined fraud columns we can use rules to estimate what data looks fraudualent
# MAGIC * Build rules based model to predict fradualent activity 
# MAGIC * Create a table with these estimates
# MAGIC * Use that new table to build a ML model using wider features
# MAGIC * Implement that ML model with the stream

# COMMAND ----------

# MAGIC %md ## Write Algorithm
# MAGIC Maybe this is just a function that can be applied later.

# COMMAND ----------

# DBTITLE 1,Create Pyfunc Model
from mlflow.pyfunc import PythonModel


class RulesBasedModel(PythonModel):

    def __init__(self, PriceDifferenceThreshold, ItemCountThreshold):
        self.PriceDifferenceThreshold = PriceDifferenceThreshold
        self.ItemCountThreshold = ItemCountThreshold

    def predict(self, context, data):
        return pd.DataFrame(data['PriceDifference'].apply(lambda x: 'True' if x > 10 else 'False'))

# COMMAND ----------

# DBTITLE 1,Save pyfunc model
import mlflow.pyfunc

with mlflow.start_run(run_name="RulesBasedModel") as run:
  
  pyfunc_model = RulesBasedModel(PriceDifferenceThreshold=200, ItemCountThreshold=2)
  
  mlflow.pyfunc.log_model(python_model=pyfunc_model, artifact_path=None)

  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# DBTITLE 1,Register Model
result=mlflow.register_model(run.info.artifact_uri, "RulesBasedModel")

# COMMAND ----------

print(f"Model version: {result.version}, Model name:{result.name}")

# COMMAND ----------

# DBTITLE 1,Transition model to Production
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

client.transition_model_version_stage(result.name, result.version, 'Production')

# COMMAND ----------

# MAGIC %md ## Apply Algorithm
# MAGIC Apply your fraud detection algorithms to the data stream.

# COMMAND ----------

model_version_uri = "models:/{model_name}/{stage}".format(model_name=result.name, stage='Production') 

# COMMAND ----------

# DBTITLE 1,Load models
# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# DBTITLE 1,Apply Rules Based Algorithm 
model_input = df.toPandas()
model_output = loaded_model.predict(model_input)
print(model_output)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Determine the number of possible fraudulent output
model_output_df = spark.createDataFrame(model_output)
display(model_output_df.groupBy('prediction').count())

# COMMAND ----------

# MAGIC %md ## Business Explanation
# MAGIC Explain to the Executive team the process and extent of fraud. Extra points for explanatory data plots.

# COMMAND ----------


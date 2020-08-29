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

from pyspark.sql.functions import *

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data")

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



# COMMAND ----------

# MAGIC %md ## Planning
# MAGIC Plan how your algorithm will find the fraud trends in real-time.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Write Algorithm
# MAGIC Maybe this is just a function that can be applied later.

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# COMMAND ----------

dataset = (df
           .withColumn('PriceDifference', col('ListPrice') - col('ChargedPrice'))
           .select('ItemCount', 'PriceDifference')
          )

# COMMAND ----------

fields = ['ItemCount', 'PriceDifference']

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=fields,
    outputCol="features")

dataset_assembled = assembler.transform(dataset)

# COMMAND ----------

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset_assembled)

# Make predictions
predictions = model.transform(dataset_assembled)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

# method 2: Scree plot
cost = list()
for k in range(2,15):
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(dataset_assembled)

    # Make predictions
    predictions = model.transform(dataset_assembled)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    cost.append(silhouette)
    print(f"k: {k}, cost: {silhouette}")

# COMMAND ----------

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset_assembled)

# Make predictions
predictions = model.transform(dataset_assembled)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

fields

# COMMAND ----------

display(predictions)

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import *

# UDF for converting column type from vector to double type
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

assembler = VectorAssembler(inputCols=[field],outputCol=field+"_Vect")

# MinMaxScaler Transformation
scaler = MinMaxScaler(inputCol=field+"_Vect", outputCol=field+"_Scaled")

# Pipeline of VectorAssembler and MinMaxScaler
pipeline = Pipeline(stages=[assembler, scaler])

# Fitting pipeline on dataframe
predictions_scaled_df = pipeline.fit(predictions).transform(predictions).withColumn(field+"_Scaled", unlist(field+"_Scaled")).drop(field+"_Vect")

# COMMAND ----------

display(predictions_scaled_df)

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# COMMAND ----------

kIdx = np.argmax(cost)

fig, ax = plt.subplots()
plt.plot(range(2,15), cost, 'b*-')
plt.plot(range(2,15)[kIdx], cost[kIdx], marker='o', markersize=12, 
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.xlim(1, plt.xlim()[1])
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Scores for k-means clustering')
# Uncomment the next line
display(fig)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Apply Algorithm
# MAGIC Apply your fraud detection algorithms to the data stream.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Business Explanation
# MAGIC Explain to the Executive team the process and extent of fraud. Extra points for explanatory data plots.

# COMMAND ----------


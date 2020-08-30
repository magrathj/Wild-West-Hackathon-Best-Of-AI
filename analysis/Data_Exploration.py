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
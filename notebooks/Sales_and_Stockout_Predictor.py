# Databricks notebook source
# MAGIC %md # Welcome to Quest #1 of Challenge 2.  
# MAGIC 
# MAGIC This notebook will comprise all of Quest #1. We suggest that you steps should be research into facebook prophet... etc. (or another library of your choosing), exploratory data analysis, planning, model training, model predictions, stock out predictions, and finally business explanation. If you are not completely confident after researching, we also suggest that once you have a plan of attack to complete the quest that you send us you plan. We can help fill in the gaps.
# MAGIC 
# MAGIC Adventure Works is trying to decide where they will place their warehouses. They are asking you, to deduce from the data, what is the smallest region type (state, region, division) where they can predict stock-out from. They realize that specific items may need to be rolled up into categories. Remember that overstocking is detremental to profits. 
# MAGIC 
# MAGIC This quest entails suggesting weekly item stock numbers per region by making daily item sales predictions. It is up to you to decide at which granularity you will make predictions on items and how large of a region you will make stock suggestions. Keep this in mind as you are looking through the data.
# MAGIC 
# MAGIC This quest will be judged by accuracy of daily predictions, stockout/overstock ratio per week, granularity of item and region, as well as your business explanation.

# COMMAND ----------

df00 = spark.table("export__6__1_csv")

# COMMAND ----------

df00.count()

# COMMAND ----------

# MAGIC %md ## Load Historical Data
# MAGIC Take a quick look at the data, so that you can start developing an understanding as you research.

# COMMAND ----------

# Import Libraries
from pyspark.sql import functions as F
import pandas as pd
from fbprophet import Prophet
import logging

# Silence some annoying output due to using python in databricks
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df00 = spark.table("historical_data")

# COMMAND ----------

display(df00)

# COMMAND ----------

df01 = spark.table("product")

# COMMAND ----------

display(df01)

# COMMAND ----------

# MAGIC %md ## Research
# MAGIC 
# MAGIC - Prophet https://facebook.github.io/prophet/docs/quick_start.html#python-api
# MAGIC - Stockout https://en.wikipedia.org/wiki/Stockout
# MAGIC - Exploratory Data Analysis https://towardsdatascience.com/exploratory-data-analysis-eda-with-pyspark-on-databricks-e8d6529626b1

# COMMAND ----------

# MAGIC %md ## Planning
# MAGIC 
# MAGIC Write out your plan here.

# COMMAND ----------

# MAGIC %md ...

# COMMAND ----------

# MAGIC %md ## Model Training
# MAGIC Train your model on historical data. Don't spend too much time here initially. Go end-to-end, then iterate.

# COMMAND ----------

# Note: pair down spark df and collect into pandas df #
for_pandas_00 = df00.select('date','ItemCount')

# prepare for prophet
for_pandas_01 = for_pandas_00.groupBy('date').agg(F.sum('ItemCount').alias('ItemCountSum')).sort('date')

# to pandas df
pandas_df_00 = for_pandas_01.toPandas()

# rename header to work with prophet
pandas_df_00 = pandas_df_00.rename(columns={'date': 'ds', 'ItemCountSum': 'y'})

pandas_df_00.head(10)

# COMMAND ----------

help(Prophet)

# COMMAND ----------

m = Prophet(daily_seasonality=True)
m.fit(pandas_df_00)

# COMMAND ----------

future = m.make_future_dataframe(periods=365)
future.tail()

# COMMAND ----------

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE TEST_DB;

# COMMAND ----------

sample_table = spark.createDataFrame(forecast)
sample_table.write.format('delta').mode("overwrite").saveAsTable('TEST_DB.forecast')

# COMMAND ----------

fig1 = m.plot(forecast)

# COMMAND ----------

# MAGIC %md ## Model Predictions
# MAGIC Attach your predictions to this daily stream data and calculate MSE per day.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Stock Out Predictions
# MAGIC Use your predictions to suggest item stock per region and calculate stockout/overstock ratio per week.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Business Explanation
# MAGIC Give a few paragraph explanation of how Adventure Works should operate their warehouse stock shipments. Extra points for explanatory data plots.

# COMMAND ----------


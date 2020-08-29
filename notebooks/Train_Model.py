# Databricks notebook source
# MAGIC %md # Welcome to Quest #3 of Challenge 2.   

# COMMAND ----------

# MAGIC %md ## Load Historical Data
# MAGIC Take a quick look at the data, so that you can start developing an understanding as you research.

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.table("historical_data")
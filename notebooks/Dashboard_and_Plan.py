# Databricks notebook source
# MAGIC %md 
# MAGIC # Welcome to the Wild West Hackathon

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.read.table("historical_data")

# COMMAND ----------

display(df)

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.read.table("PossibleFraud")

# COMMAND ----------

display(df)

# COMMAND ----------


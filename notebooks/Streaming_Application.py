# Databricks notebook source
# DBTITLE 1,Get schema of the table
# Note: CurrentDatetime is null, use date instead #
df_static = spark.read.table("historical_data")

# COMMAND ----------

# DBTITLE 1,Set max records to 1, so that the stream processes them one record at a time
spark.conf.set("spark.sql.files.maxRecordsPerFile", 1)

# COMMAND ----------

# DBTITLE 1,Write them to a tmp location
df_static.write.format("parquet").mode("overwrite").save('/mnt/streaming_data')

# COMMAND ----------

# DBTITLE 1,Read in the stream
streaming_df = (spark.readStream.format("parquet").option('maxFilesPerTrigger', 1).load('/mnt/streaming_data'))

# COMMAND ----------

streaming_df.count()

# COMMAND ----------

(streaming_df
  .format("memory")
  .queryName("tableName")
  .start()
)
# Databricks notebook source
dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get model name
name=dbutils.widgets.getArgument('modelName', 'RulesBasedModel')

# COMMAND ----------

# DBTITLE 1,Get current stage you want to transition
current_stage=dbutils.widgets.getArgument('modelCurrentStage', 'Staging')

# COMMAND ----------

# DBTITLE 1,Get next stage you want to transition to
next_stage=dbutils.widgets.getArgument('modelNextStage', 'Production')

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

# COMMAND ----------

version=client.get_latest_versions(name=name, stages=[current_stage])

# COMMAND ----------

client.transition_model_version_stage(name, version[0].version, next_stage)
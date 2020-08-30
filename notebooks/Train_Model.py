# Databricks notebook source
# MAGIC %md ## Load Historical Data

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df = spark.table("DataWithFraudPredictions").withColumn('label', when(col('prediction')=='True', lit(1)).otherwise(lit(0))).drop('prediction')

# COMMAND ----------

# DBTITLE 1,Split training and test sets
# Split our dataset between training and test datasets
(train, test) = df.randomSplit([0.8, 0.2], seed=12345)

# COMMAND ----------

# DBTITLE 1,Define columns
# Encodes a string column of labels to a column of label indices
indexer = StringIndexer(inputCol = "RepName", outputCol = "RepNameIndexed")

# VectorAssembler is a transformer that combines a given list of columns into a single vector column
va = VectorAssembler(inputCols = ['RepNameIndexed', 
                                 'MLMountainTire',
                                 'LLRoadFrameRed48',
                                 'Touring1000Blue54',
                                 'AllPurposeBikeStand',
                                 'Mountain400WSilver42',
                                 'PatchKit8Patches',
                                 'Road650Red58',
                                 'HLMountainRearWheel',
                                 'LLMountainFrameSilver44',
                                 'FenderSetMountain',
                                 'MLRoadFrameRed44',
                                 'HLMountainFrameSilver48',
                                 'Road650Black62',
                                 'MountainBikeSocksM',
                                 'HitchRack4Bike',
                                 'LLRoadTire',
                                 'Road750Black48',
                                 'TouringTire',
                                 'MLRoadHandlebars',
                                 'MLCrankset',
                                 'HLRoadFrameBlack62',
                                 'LLRoadFrameRed44',
                                 'HLRoadFrameBlack52',
                                 'Mountain500Black40',
                                 'Road250Black44',
                                 'Mountain400WSilver38',
                                 'HLTouringFrameYellow46',
                                 'ItemCount',
                                 'LLTouringFrameYellow54',
                                 'Road550WYellow44',
                                 'LLMountainFrameBlack44',
                                 'MLMountainPedal',
                                 'Road250Red44',
                                 'WomensMountainShortsM',
                                 'MLMountainFrameBlack48',
                                 'LLFork',
                                 'MLRoadFrameRed52',
                                 'LLRoadPedal',
                                 'HLRoadFrameRed44',
                                 'Road250Red52',
                                 'FrontDerailleur',
                                 'TouringPanniersLarge',
                                 'LLHeadset',
                                 'ListPrice',
                                 'ClassicVestL',
                                 'RoadBottleCage',
                                 'LLTouringFrameYellow58',
                                 'HLBottomBracket',
                                 'Road550WYellow48',
                                 'Road650Red62',
                                 'Sport100HelmetRed',
                                 'Mountain300Black48',
                                 'MensBibShortsL',
                                 'LLCrankset',
                                 'Road250Red48',
                                 'HLMountainTire',
                                 'Touring1000Yellow50',
                                 'HalfFingerGlovesM',
                                 'Mountain500Black48',
                                 'ClassicVestS',
                                 'Touring3000Yellow54',
                                 'HLRoadFrameBlack58',
                                 'MensSportsShortsS',
                                 'HLTouringFrameYellow60',
                                 'MLRoadFrameWYellow38',
                                 'MLMountainRearWheel',
                                 'MLMountainFrameBlack38',
                                 'MLMountainFrameWSilver42',
                                 'FullFingerGlovesS',
                                 'MLMountainFrameWSilver38',
                                 'Mountain500Silver44',
                                 'HLMountainFrameSilver44',
                                 'LLTouringSeatSaddle',
                                 'Road650Red52',
                                 'HeadlightsDualBeam',
                                 'Mountain500Silver42',
                                 'LLMountainFrontWheel',
                                 'LongSleeveLogoJerseyL',
                                 'HLRoadFrameRed58',
                                 'Mountain100Black44',
                                 'Road350WYellow44',
                                 'Touring3000Blue54',
                                 'Road250Black48',
                                 'Touring1000Yellow54',
                                 'LLRoadFrameBlack62',
                                 'RacingSocksL',
                                 'RearDerailleur',
                                 'HLMountainHandlebars',
                                 'MLRoadFrameWYellow40',
                                 'LLRoadFrontWheel',
                                 'HLMountainFrameBlack42',
                                 'MountainTireTube',
                                 'LLTouringFrameYellow50',
                                 'Road550WYellow40',
                                 'MLFork',
                                 'Mountain100Silver44',
                                 'Road650Black52',
                                 'Touring2000Blue60',
                                 'MountainBikeSocksL',
                                 'Sport100HelmetBlue',
                                 'MLMountainFrameWSilver40',
                                 'LLTouringFrameBlue44',
                                 'MLRoadFrontWheel',
                                 'FullFingerGlovesL',
                                 'WomensTightsS',
                                 'LLRoadFrameRed52',
                                 'Road650Black58',
                                 'LLTouringHandlebars',
                                 'LLMountainSeatSaddle',
                                 'Road350WYellow48',
                                 'LLMountainFrameSilver40',
                                 'HLHeadset',
                                 'Mountain100Silver48',
                                 'BikeWashDissolver',
                                 'Touring3000Blue62',
                                 'Road350WYellow42',
                                 'Mountain100Black38',
                                 'LLMountainPedal',
                                 'MensSportsShortsL',
                                 'MLRoadFrameRed60',
                                 'Mountain100Silver42',
                                 'Road750Black52',
                                 'Mountain200Black38',
                                 'Mountain500Silver48',
                                 'Road150Red56',
                                 'Touring3000Yellow50',
                                 'Touring1000Yellow46',
                                 'MLHeadset',
                                 'Mountain500Black44',
                                 'MLRoadFrameWYellow48',
                                 'LLRoadFrameRed60',
                                 'WomensTightsM',
                                 'ClassicVestM',
                                 'LLTouringFrameBlue58',
                                 'Mountain300Black38',
                                 'TouringRearWheel',
                                 'HLTouringFrameYellow54',
                                 'HalfFingerGlovesS',
                                 'Touring1000Yellow60',
                                 'LongSleeveLogoJerseyM',
                                 'MensBibShortsM',
                                 'Road150Red52',
                                 'HLFork',
                                 'MLRoadFrameWYellow44',
                                 'LLTouringFrameYellow44',
                                 'LLMountainFrameBlack40',
                                 'HLMountainFrameBlack46',
                                 'Touring1000Blue60',
                                 'Sport100HelmetBlack',
                                 'FullFingerGlovesM',
                                 'MensSportsShortsXL',
                                 'HLMountainSeatSaddle',
                                 'Chain',
                                 'LLMountainHandlebars',
                                 'MLRoadRearWheel',
                                 'Mountain100Silver38',
                                 'MLMountainFrameBlack40',
                                 'Road750Black44',
                                 'LLRoadFrameBlack44',
                                 'Road450Red52',
                                 'LLMountainFrameBlack52',
                                 'ShortSleeveClassicJerseyXL',
                                 'HLMountainFrameBlack38',
                                 'Road150Red48',
                                 'LLRoadHandlebars',
                                 'AWCLogoCap',
                                 'HLTouringFrameBlue50',
                                 'Touring1000Blue46',
                                 'MLMountainFrontWheel',
                                 'Touring3000Yellow62',
                                 'HLRoadPedal',
                                 'TouringTireTube',
                                 'Mountain100Black48',
                                 'ChargedPrice',
                                 'MensSportsShortsM',
                                 'Mountain300Black40',
                                 'LLBottomBracket',
                                 'MountainPump',
                                 'MLRoadTire',
                                 'MLMountainFrameBlack44',
                                 'MLRoadFrameRed48',
                                 'HLTouringFrameYellow50',
                                 'Road250Black52',
                                 'Touring1000Blue50',
                                 'MLMountainSeatSaddle',
                                 'Road450Red44',
                                 'Road150Red62',
                                 'LLTouringFrameYellow62',
                                 'MLMountainHandlebars',
                                 'LLTouringFrameBlue62',
                                 'HLRoadFrameRed56',
                                 'Mountain200Black42',
                                 'LLRoadFrameBlack58',
                                 'Road250Black58',
                                 'TouringFrontWheel',
                                 'Road550WYellow38',
                                 'LLTouringFrameBlue54',
                                 'HLRoadTire',
                                 'ShortSleeveClassicJerseyL',
                                 'LLMountainFrameSilver42',
                                 'Road650Black44',
                                 'HLMountainFrameSilver46',
                                 'HLRoadSeatSaddle',
                                 'HLMountainPedal',
                                 'Touring3000Blue58',
                                 'Mountain400WSilver46',
                                 'Touring2000Blue54',
                                 'Mountain200Silver46',
                                 'Road450Red48',
                                 'RacingSocksM',
                                 'Mountain400WSilver40',
                                 'Touring3000Yellow44',
                                 'HLRoadFrontWheel',
                                 'HLTouringSeatSaddle',
                                 'Road650Black48',
                                 'LLRoadFrameBlack48',
                                 'Minipump',
                                 'Road550WYellow42',
                                 'Mountain200Silver42',
                                 'HLRoadFrameBlack44',
                                 'Road650Red60',
                                 'WomensTightsL',
                                 'HLRoadHandlebars',
                                 'HLTouringHandlebars',
                                 'HLTouringFrameBlue54',
                                 'Touring2000Blue50',
                                 'TouringPedal',
                                 'Road650Red44',
                                 'Road650Red48',
                                 'FrontBrakes',
                                 'Touring3000Yellow58',
                                 'Road150Red44',
                                 'MLRoadFrameWYellow42',
                                 'TaillightsBatteryPowered',
                                 'Mountain200Silver38',
                                 'HLRoadFrameBlack48',
                                 'Mountain500Silver40',
                                 'Mountain500Black52',
                                 'Mountain300Black44',
                                 'RoadTireTube',
                                 'ShortSleeveClassicJerseyS',
                                 'MountainBottleCage',
                                 'MLRoadSeatSaddle',
                                 'Road750Black58',
                                 'Mountain100Black42',
                                 'MensBibShortsS',
                                 'LLRoadRearWheel',
                                 'Mountain200Black46',
                                 'LLTouringFrameBlue50',
                                 'LongSleeveLogoJerseyS',
                                 'WomensMountainShortsL',
                                 'Mountain500Black42',
                                 'MLBottomBracket',
                                 'RearBrakes',
                                 'MLRoadFrameRed58',
                                 'HLMountainFrameBlack44',
                                 'LLMountainFrameBlack48',
                                 'LLMountainTire',
                                 'HLMountainFrameSilver38',
                                 'HLRoadRearWheel',
                                 'CableLock',
                                 'LLRoadFrameRed58',
                                 'Road250Red58',
                                 'HLTouringFrameBlue60',
                                 'LLRoadSeatSaddle',
                                 'Touring3000Blue44',
                                 'LLRoadFrameBlack60',
                                 'MLTouringSeatSaddle',
                                 'HLRoadFrameRed62',
                                 'HLMountainFrameBlack48',
                                 'LLRoadFrameRed62',
                                 'HLMountainFrameSilver42',
                                 'HLRoadFrameRed52',
                                 'HalfFingerGlovesL',
                                 'MLMountainFrameWSilver46',
                                 'HLTouringFrameBlue46',
                                 'LLMountainFrameSilver52',
                                 'WomensMountainShortsS',
                                 'MLRoadPedal',
                                 'LLRoadFrameBlack52',
                                 'HLRoadFrameRed48',
                                 'PreviousPrice',
                                 'StandardCost',
                                 'Touring3000Blue50',
                                 'LLMountainRearWheel',
                                 'LLMountainFrameBlack42',
                                 'WaterBottle30oz',
                                 'Road650Black60',
                                 'Road350WYellow40',
                                 'Mountain500Silver52',
                                 'HLMountainFrontWheel',
                                 'HeadlightsWeatherproof',
                                 'Road450Red60',
                                 'LLMountainFrameSilver48',
                                 'ShortSleeveClassicJerseyM',
                                 'Road450Red58',
                                 'Touring2000Blue46',
                                 'HydrationPack70oz',
                                 'HLCrankset'], outputCol = "features")

# COMMAND ----------

# DBTITLE 1,Create Decision Tree Classifier
# Using the DecisionTree classifier model
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features", seed = 54321, maxDepth = 5)

# Create our pipeline stages

pipeline = Pipeline(stages=[indexer, va, dt])

# View the Decision Tree model (prior to CrossValidator)
dt_model = pipeline.fit(train)

# COMMAND ----------

# DBTITLE 1,Create Evaluators
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use BinaryClassificationEvaluator to evaluate our model
evaluatorPR = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderPR")
evaluatorAUC = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderROC")

# COMMAND ----------

# DBTITLE 1,Create confusion matrix df
from pyspark.sql.functions import lit, expr, col, column

# Confusion matrix template
cmt = spark.createDataFrame([(1, 0), (0, 0), (1, 1), (0, 1)], ["label", "prediction"])
cmt.createOrReplaceTempView("cmt")

# COMMAND ----------

# DBTITLE 1,Create experiment, log metrics and model
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


with mlflow.start_run(run_name="FraudMLModel") as run:
  
    params = {"maxDepth": [5, 10, 15, 20, 25, 30], "maxBins": [30, 40, 50]}
    
    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    
    # Build the grid of different parameters
    paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15, 20, 25, 30]) \
    .addGrid(dt.maxBins, [30, 40, 50]) \
    .build()

    # Build out the cross validation
    crossval = CrossValidator(estimator = dt,
                              estimatorParamMaps = paramGrid,
                              evaluator = evaluatorPR,
                              numFolds = 3)  
    # Build the CV pipeline
    pipelineCV = Pipeline(stages=[indexer, va, crossval])

    # Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
    cvModel_u = pipelineCV.fit(train)
    
    # Build the best model (training and test datasets)
    train_pred = cvModel_u.transform(train)
    test_pred = cvModel_u.transform(test)

    # Evaluate the model on training datasets
    pr_train = evaluatorPR.evaluate(train_pred)
    auc_train = evaluatorAUC.evaluate(train_pred)

    # Evaluate the model on test datasets
    pr_test = evaluatorPR.evaluate(test_pred)
    auc_test = evaluatorAUC.evaluate(test_pred)

    # Print out the PR and AUC values
    print("PR train:", pr_train)
    print("AUC train:", auc_train)
    print("PR test:", pr_test)
    print("AUC test:", auc_test)
    
    mlflow.log_metrics({"pr_train": pr_train, "AUC_train": auc_train, "pr_test":pr_test, "AUC_test":auc_test})
    
    # Log model
    mlflow.spark.log_model(cvModel_u, "model")
    
    # Create temporary view for test predictions
    test_pred.createOrReplaceTempView("test_pred")

    # Create test predictions confusion matrix
    test_pred_cmdf = spark.sql("select a.label, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select label, prediction, count(1) as count from test_pred group by label, prediction) b on b.label = a.label and b.prediction = a.prediction order by a.label desc, a.prediction desc")

    ## Write csv from stats dataframe
    test_pred_cmdf.toPandas().to_csv('confusion.csv')

    ## Log CSV to MLflow
    mlflow.log_artifact('confusion.csv')
    
    # View confusion matrix
    display(test_pred_cmdf)
    
    run_id = mlflow.active_run().info.run_id
    artifact_path = "model"
    
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

# COMMAND ----------

# DBTITLE 1,Register Model
result = mlflow.register_model(
    model_uri,
    "CrossValidatorModel"
)

# COMMAND ----------


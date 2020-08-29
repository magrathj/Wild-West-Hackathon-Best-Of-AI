# Databricks notebook source
# MAGIC %md # Welcome to Quest #2 of Challenge 2.  
# MAGIC 
# MAGIC This notebook will comprise all of Quest #2. We suggest that your steps should be research, exploratory data analysis, planning, model training, recommendation, and finally business explanation. If you are not completely confident after researching, we also suggest that once you have a plan of attack to complete the quest that you send us you plan. We can help fill in the gaps.
# MAGIC 
# MAGIC The Adventure Works marketing team needs help making item recommendations. They have two scenarios where recommendations would be useful. First, they would like to send a marketing campaign to customers after each purchase order. Second, based off of a new customer profile, suggest items they might want to buy. The Adventure Works website has a "choose the bike that suits you" game, in which they are able to collect information about incoming customers (if you were wondering how they have so much data on new customers).  
# MAGIC 
# MAGIC This quest entails making a item recommendation engine and applying it to the two different marketing scenarios. 
# MAGIC 
# MAGIC You will be judged by accuracy of both recommendation engines, as well as your business explanation.

# COMMAND ----------

# MAGIC %md ## Load Historical Data
# MAGIC Take a quick look at the data, so that you can start developing an understanding as you research.

# COMMAND ----------

# Note: CurrentDatetime is null, use date instead #
df00 = spark.table("historical_data")

# COMMAND ----------

display(df00)

# COMMAND ----------

# MAGIC %md ## Research
# MAGIC - Pyspark Recommendation Engine https://medium.com/analytics-vidhya/crafting-recommendation-engine-in-pyspark-a7ca242ad40a
# MAGIC - Exploratory Data Analysis https://towardsdatascience.com/exploratory-data-analysis-eda-with-pyspark-on-databricks-e8d6529626b1

# COMMAND ----------

# MAGIC %md ## Planning
# MAGIC 
# MAGIC Write out your plan here.

# COMMAND ----------

# MAGIC %md ...

# COMMAND ----------

# MAGIC %md ## Model(s) Training
# MAGIC Train your model on historical data. Don't spend too much time here initially. Go end-to-end, then iterate.

# COMMAND ----------

# It is going to take some work to get the data into als format, if that is the algorithms that you are going to use, message David Wood if you want some pointers

# COMMAND ----------

# MAGIC %md ## Recommendations for Existing Customers
# MAGIC Apply recommendations to the existing customer stream.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Recommendations for New Customers
# MAGIC Apply recommendations to the new customer stream.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Business Explanation
# MAGIC Give a few paragraph explanation of how your recommender engines work and how the results should be interpreted. Extra points for explanatory data plots.

# COMMAND ----------


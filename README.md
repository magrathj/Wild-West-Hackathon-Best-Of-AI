# Wild-West-Hackathon-Best-Of-AI
Wild West Hackathon v2 - using databricks, mflow and azure devops to build automated deployment of fraud detection machine learning models

## Hackathon Challenge Brief

Welcome to Quest #3 of Challenge 2.

This notebook will comprise all of Quest #3. We suggest that your steps should be data exploration, planning, write algorithm, apply algorithm, and finally business explanation. If you are not completely confident after researching, we also suggest that once you have a plan of attack to complete the quest that you send us you plan. We can help fill in the gaps.

The Adventure Works executive and finance teams have had some unexpected financial results. They suspect foul play! You have been hired to find examples of fraud and report when it happens in real-time.

1) Adventure works only receives a total charge for each customer order from each rep. This is because the reps have the freedom to give discretionary discounts to customers if they think it will sway them to but product. The executive team thinks that there is rampant fraud amoung the reps. They suspect that the reps are adding items to a customers order and then taking that item home with them. Adventure works needs you to establish a pattern in finding this behavior then compile a list, moving forward, of fraud examples.

2) Adventure Works let go of their systems administrator a few years back. The systems admin was very unhappy with this and the executive team heard from a co-worker that the sys admin is a known hacker. The executive team is scared that the sys admin left something behind in the system. If you can find the discrepancy in real time then it will be easier for engineers to track down the hack.

Note there exists only two types of fraud present in the data following the scenario above should make them very evident.

This quest entails identifying the fraud trends and writing an algorithm to find fraud in real time.

You will be judged by accuracy in finding real-time fraud, as well as your business explanation.


## Initial Setup to connect VS Code to Databricks Env


### Create Token
![Create token](./images/create-token.PNG)

### Set up databricks cli
```
    databricks configure --token --profile hackathon-v2
```

### View workspace
```
    databricks workspace list --profile hackathon-v2
```

### Export current notebooks
```
    ./export_notebooks.sh
```

## Challenge Findings

### Exporation

Analysing the differences between listing prices, cost price and the actual price it was sold for over a period of time.
![PriceDifferenceOverTime](./images/Exploration_trends_over_time.PNG)


![Reps](./images/Exploration_analysing_reps_customer_counts.PNG)

Analysing the differences between listing prices and the actual price it was sold for.
![PriceDifference](./images/Exploration_analysing_difference_between_listing_selling_pricing.PNG)

### Planning

* Analyse what potential columns would identicate fraud - as we dont have defined fraud columns we can use rules to estimate what data looks fraudualent
* Build rules based model to predict fradualent activity
* Create a table with these estimates
* Use that new table to build a ML model using wider features
* Implement that ML model with the stream


### Business Explanation

Business goals:
* Reduce expense base by identifying as much possible fraud as possible
* Deploy quickly and iterate into production as soon as possible to get a ROI for the data science activities

Identifying as much fraud as possible:
1. Build rules based model
2. Build a pipeline, so as to use that rules based model to train a machine learning model
3. Deploy both the rules based and machine learning model into a production streaming application, so as to identify as many possible counts of fraud as possible

Proving the ROI of data science
1. Build pipeline to allow iteration
2. Use azure devops to automate that pipeline 
3. Iterate and allows changes to propogate through the pipeline. This will allow for actionalble insights to be drawn straight away, even if they are not 100% correct. 


#### Using MLflow to deliver models into production setting quickly

![FraudMLModel_MLflow_Runs](./images/FraudMLModel_MLflow_Runs.PNG)

![MLflow_Comparing_Runs](./images/MLflow_Comparing_Runs.PNG)

![Register_Models](./images/Register_Models.PNG)

![Saved_model_artifacts](./images/Saved_model_artifacts.PNG)

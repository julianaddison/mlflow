# Utilising MLflow for Experiment Tracking
For the purpose of the experiment, a time series model is trained. An XGBoost is tuned with several different hyperparameters using time series cross validation on SkLearn. All metrics are logged in the MLflow server. 

## Setup
1. Setup MLflow in Databricks:
Follow the [link](https://community.cloud.databricks.com/login.html), and complete the signing up process.

2. Login to databricks which should bring you to this page. On the left pane, go to `Machine Learning` > `Experiments` then `Create Experiment` on the top right.
![Landing Page](https://github.com/julianaddison/mlflow/blob/main/images/experiment_landing_page.png)

3. Setup an experiment on the server called `call_forecasting`
![Create Exeperiment Page](https://github.com/julianaddison/mlflow/blob/main/images/create_experiment.png)

## Run

## Expected Results
The experiment metrics are logged in MLflow as the model trains. `Run Name` can be configured in the python script to better reflect each run. For each model the MSE, MAE, and MAPE values are logged and can be visualised in the Chart tab. The `Run Name` = best_model is the model with the lowest MSE. All model artifacts including dependency files are created e.g. conda.yaml, requirements.txt, etc.
https://github.com/julianaddison/mlflow/assets/32608788/97bd5648-87df-4051-b171-e2c72c478193



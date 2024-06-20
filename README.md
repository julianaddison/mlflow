# MLflow for Experiment Tracking
### What is MLflow?
MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It simplifies and accelerates the process of developing and deploying machine learning models by providing a suite of tools for experiment tracking, model packaging, and model deployment.

Databricks CE MLflow service is a free MLflow tracking server provided by Databricks. The vast majority of MLflow functionality is supported (with the notable exception that you cannot create serving endpoints on CE, so deployment of models is not supported). A self-managed or local MLflow server can be setup [see instructions](https://mlflow.org/docs/latest/getting-started/running-notebooks/index.html).

### Experiment
For the purpose of the experiment, a time series call forecasting model is trained. XGBoost is tuned with several different hyperparameters using time series cross validation on SKLearn. All metrics are logged in the Databricks Community Edition (CE) which has MLflow service.

## Prerequisites & Setup
1. Setup MLflow in Databricks:
Follow the [link](https://community.cloud.databricks.com/login.html), and complete the signing up process.

2. Login to databricks which should bring you to this page. On the left pane, go to `Machine Learning` > `Experiments` then `Create Experiment` on the top right.
![Landing Page](https://github.com/julianaddison/mlflow/blob/main/images/experiment_landing_page.png)

3. Setup an experiment on the server called `call_forecasting`
![Create Exeperiment Page](https://github.com/julianaddison/mlflow/blob/main/images/create_experiment.png)

## Run
### Google Collab
1. Setup `DatabricksUserName` in Google Collab Secrets. Refer to this [article](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75) on how to store the keys.
2. Input your databricks username and password when prompted after executing the cell `!databricks configure --host https://community.cloud.databricks.com/`
3. Run the remaining cells and observe the [experiment results](#expected-results) in the Experiment window in Databricks

## Expected Results
The experiment metrics are logged in MLflow as the model trains. `Run Name` can be configured in the python script to better reflect each run. For each model the MSE, MAE, and MAPE values are logged and can be visualised in the Chart tab. The `Run Name` = best_model is the model with the lowest MSE. All model artifacts including dependency files are created e.g. conda.yaml, requirements.txt, etc.

https://github.com/julianaddison/mlflow/assets/32608788/97bd5648-87df-4051-b171-e2c72c478193



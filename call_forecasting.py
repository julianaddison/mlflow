import os
from itertools import product
import pprint
import requests
from urllib.parse import urljoin

import pandas as pd
import datetime
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import mlflow


def download_github_folder(repo_url, folder_path, local_path='data'):
  if not os.path.exists(local_path):
      os.makedirs(local_path)
  
  api_url = repo_url.replace('https://github.com/', 'https://api.github.com/repos/')
  contents_url = urljoin(api_url, f'contents/{folder_path}')

  # Add headers to handle potential API rate limits and authentication
  headers = {
      'Accept': 'application/vnd.github.v3+json',
      # 'Authorization': 'token YOUR_PERSONAL_ACCESS_TOKEN'  # Uncomment if using a personal access token
  }
  
  response = requests.get(contents_url, headers=headers)
  print(response)
  if response.status_code == 200:
      for file_info in response.json():
          file_path = file_info['path']
          download_url = file_info['download_url']
          local_file_path = os.path.join(local_path, os.path.basename(file_path))
          
          with requests.get(download_url, stream=True) as r:
              r.raise_for_status()
              with open(local_file_path, 'wb') as f:
                  for chunk in r.iter_content(chunk_size=8192):
                      f.write(chunk)
          print(f'Downloaded {local_file_path}')
  else:
      print(f'Failed to retrieve contents: {response.status_code}')
  return


def data_preparation(case_data_calls_fp, case_data_reservations_fp, window, prediction_offset):
  # load data
  calls_df = pd.read_csv(case_data_calls_fp, sep=';')
  reservations_df = pd.read_csv(case_data_reservations_fp, sep=';')

  # convert data types
  calls_df['date'] = pd.to_datetime(calls_df['date'], format="%d-%m-%Y")
  reservations_df['date'] = pd.to_datetime(reservations_df['date'], format="%d-%m-%Y")

  # create copy of calls dataframe and sort by date
  _calls_df = calls_df.copy()
  _calls_df.sort_values(by=['date'],ascending=True)

  # Adding additional date-related features
  _calls_df['quarter'] = _calls_df['date'].dt.quarter
  _calls_df['month'] = _calls_df['date'].dt.month
  _calls_df['year'] = _calls_df['date'].dt.year
  _calls_df['weekofyear'] = _calls_df['date'].dt.isocalendar().week
  _calls_df['calls_7_day_lag'] = _calls_df['calls'].shift(7)
  _calls_df['calls_7_day_lag_mean'] = _calls_df['calls'].rolling(window=7).mean()
  _calls_df['calls_7_day_lag_std'] = _calls_df['calls'].rolling(window=7).std()
  _calls_df['calls_7_day_lag_max'] = _calls_df['calls'].rolling(window=7).max()
  _calls_df['calls_7_day_lag_min'] = _calls_df['calls'].rolling(window=7).min()
  _calls_df['calls_14_day_lag'] = _calls_df['calls'].shift(14)
  _calls_df['calls_14_day_lag_mean'] = _calls_df['calls'].rolling(window=14).mean()
  _calls_df['calls_14_day_lag_std'] = _calls_df['calls'].rolling(window=14).std()
  _calls_df['calls_14_day_lag_max'] = _calls_df['calls'].rolling(window=14).max()
  _calls_df['calls_14_day_lag_min'] = _calls_df['calls'].rolling(window=14).min()
  _calls_df['calls_21_day_lag'] = _calls_df['calls'].shift(21)
  _calls_df['calls_21_day_lag_mean'] = _calls_df['calls'].rolling(window=21).mean()
  _calls_df['calls_21_day_lag_std'] = _calls_df['calls'].rolling(window=21).std()
  _calls_df['calls_21_day_lag_max'] = _calls_df['calls'].rolling(window=21).max()
  _calls_df['calls_21_day_lag_min'] = _calls_df['calls'].rolling(window=21).min()

  # Create lagged feature for total_reservations with 7,14,21 day shifts
  reservations_df['total_reservations_7_day_lag'] = reservations_df['total_reservations'].shift(7)
  reservations_df['total_reservations_7_day_lag_mean'] = reservations_df['total_reservations'].rolling(window=7).mean()
  reservations_df['total_reservations_7_day_lag_std'] = reservations_df['total_reservations'].rolling(window=7).std()
  reservations_df['total_reservations_7_day_lag_max'] = reservations_df['total_reservations'].rolling(window=7).max()
  reservations_df['total_reservations_7_day_lag_min'] = reservations_df['total_reservations'].rolling(window=7).min()
  reservations_df['total_reservations_14_day_lag'] = reservations_df['total_reservations'].shift(14)
  reservations_df['total_reservations_14_day_lag_mean'] = reservations_df['total_reservations'].rolling(window=14).mean()
  reservations_df['total_reservations_14_day_lag_std'] = reservations_df['total_reservations'].rolling(window=14).std()
  reservations_df['total_reservations_14_day_lag_max'] = reservations_df['total_reservations'].rolling(window=14).max()
  reservations_df['total_reservations_14_day_lag_min'] = reservations_df['total_reservations'].rolling(window=14).min()
  reservations_df['total_reservations_21_day_lag'] = reservations_df['total_reservations'].shift(21)
  reservations_df['total_reservations_21_day_lag_mean'] = reservations_df['total_reservations'].rolling(window=21).mean()
  reservations_df['total_reservations_21_day_lag_std'] = reservations_df['total_reservations'].rolling(window=21).std()
  reservations_df['total_reservations_21_day_lag_max'] = reservations_df['total_reservations'].rolling(window=21).max()
  reservations_df['total_reservations_21_day_lag_min'] = reservations_df['total_reservations'].rolling(window=21).min()

  # Transform the data with WINDOW days as features and a (PREDICTION_OFFSET+1)-day prediction offset
  transformed_calls_df = transform_data_with_pred_offset(_calls_df, feature_len=window, prediction_offset=prediction_offset)

  # merge transformed calls to reservations data
  calls_reservations_df = transformed_calls_df.merge(reservations_df, on='date')

  # create weekend flag
  calls_reservations_df['is_weekend'] = calls_reservations_df['weekday'] >= 6
  calls_reservations_df['is_weekend'] = calls_reservations_df['is_weekend'].astype(int)
  
  return calls_reservations_df


def transform_data_with_pred_offset(df, feature_len, prediction_offset):
  # Initialize an empty list to store the transformed data
  transformed_data = []
  window_len = feature_len + prediction_offset

  # Loop over the DataFrame to create the transformed structure
  for i in range(len(df) - window_len):
      # Create a temporary list to store one row of the new structure
      temp_row = []
      # Append the date of the 13th day
      temp_row.append(df.loc[i + window_len, 'date'])
      # Append the num_calls data for the first feature_len e.g. 6 days
      temp_row.extend(df.loc[i:i + feature_len - 1, 'calls'].tolist())
      # Append the num_calls data for the window_len e.g. 13th day if feature_len=6 and prediction_offset=7
      temp_row.append(df.loc[i + window_len, 'calls'])
      # Append the weekday of the window_len e.g. 13th day if feature_len=6 and prediction_offset=7
      temp_row.append(df.loc[i + window_len, 'weekday'])

      # Append additional date-related features for the 13th day
      temp_row.append(df.loc[i + window_len, 'quarter'])
      temp_row.append(df.loc[i + window_len, 'month'])
      temp_row.append(df.loc[i + window_len, 'year'])
      temp_row.append(df.loc[i + window_len, 'weekofyear'])
      temp_row.append(df.loc[i + window_len, 'calls_7_day_lag'])
      temp_row.append(df.loc[i + window_len, 'calls_7_day_lag_mean'])
      temp_row.append(df.loc[i + window_len, 'calls_7_day_lag_std'])
      temp_row.append(df.loc[i + window_len, 'calls_7_day_lag_max'])
      temp_row.append(df.loc[i + window_len, 'calls_7_day_lag_min'])
      temp_row.append(df.loc[i + window_len, 'calls_14_day_lag'])
      temp_row.append(df.loc[i + window_len, 'calls_14_day_lag_mean'])
      temp_row.append(df.loc[i + window_len, 'calls_14_day_lag_std'])
      temp_row.append(df.loc[i + window_len, 'calls_14_day_lag_max'])
      temp_row.append(df.loc[i + window_len, 'calls_14_day_lag_min'])
      temp_row.append(df.loc[i + window_len, 'calls_21_day_lag'])
      temp_row.append(df.loc[i + window_len, 'calls_21_day_lag_mean'])
      temp_row.append(df.loc[i + window_len, 'calls_21_day_lag_std'])
      temp_row.append(df.loc[i + window_len, 'calls_21_day_lag_max'])
      temp_row.append(df.loc[i + window_len, 'calls_21_day_lag_min'])
      # Append the temporary row to the transformed_data list
      transformed_data.append(temp_row)

  # Define the column names for the new DataFrame
  calls_name_list = ['calls_day_' + str(ind + 1) for ind in range(feature_len)]
  columns = ['date'] + calls_name_list + [f'calls_day_{window_len+1}', 'weekday'] + ['quarter', 'month', 'year', 'weekofyear',
                                                                                     'calls_7_day_lag', 'calls_7_day_lag_mean', 'calls_7_day_lag_std', 'calls_7_day_lag_max', 'calls_7_day_lag_min',
                                                                                     'calls_14_day_lag', 'calls_14_day_lag_mean', 'calls_14_day_lag_std', 'calls_14_day_lag_max', 'calls_14_day_lag_min',
                                                                                     'calls_21_day_lag', 'calls_21_day_lag_mean', 'calls_21_day_lag_std', 'calls_21_day_lag_max', 'calls_21_day_lag_min']

  # Create the new DataFrame
  transformed_df = pd.DataFrame(transformed_data, columns=columns)

  return transformed_df




def setup_train_test_data(calls_reservations_df, window, prediction_offset):
  # Feature and target variable
  forecast_day = window+prediction_offset+1
  cols_to_drop = [f'calls_day_{forecast_day}']
  add_cols_to_drop = ['total_reservations']  # 'weekday'
  cols_to_drop.extend(add_cols_to_drop)

  # drop NA rows
  calls_reservations_df = calls_reservations_df.dropna()
  calls_reservations_df.reset_index(drop=True, inplace=True)

  # Set X as features, y as target
  X = calls_reservations_df.drop(columns=cols_to_drop)
  y = calls_reservations_df[f'calls_day_{forecast_day}']

  # Ensure data is ordered by date
  X = X.sort_index()
  y = y.sort_index()

  # Define the size of the training set (e.g., 80% of the data)
  train_size = int(len(calls_reservations_df) * 0.8)

  # Split the data
  X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
  y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

  # Drop dates
  training_dates = X_train["date"]
  testing_dates = X_test["date"]
  X_train = X_train.drop(columns=['date'])
  X_test = X_test.drop(columns=['date'])

  # Verify the split
  print("Training set:")
  print(X_train.shape)
  print(y_train.shape)
  return X_train, X_test, y_train, y_test


def evaluate_model(y_test, prediction):
  mae = mean_absolute_error(y_test, prediction)
  mse = mean_squared_error(y_test, prediction)
  mape = mean_absolute_percentage_error(y_test, prediction)

  print(f"MAE: {mae}")
  print(f"MSE: {mse}")
  print(f"MAPE: {mape}")
  return mae, mse, mape


def run_best_model(X_train, y_train, X_test, y_test, best_model, best_params):
  # After all runs, evaluate the best model on the test set
  with mlflow.start_run(run_name="best_model"):
    # Train the best model on the entire training data
    best_model.fit(X_train, y_train)
    
    # Make predictions and evaluate model on Test
    y_pred = best_model.predict(X_test)
    test_mae, test_mse, test_mape = evaluate_model(y_test, y_pred)

    # Log test metrics
    mlflow.log_param("best_max_depth", best_params["max_depth"])
    mlflow.log_param("best_learning_rate", best_params["learning_rate"])
    mlflow.log_param("best_n_estimators", best_params["n_estimators"])
    mlflow.log_param("best_colsample_bytree", best_params["colsample_bytree"])
    mlflow.log_param("best_lambda", best_params["lambda"])
    mlflow.log_param("best_alpha", best_params["alpha"])
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_mape", test_mape)
    mlflow.xgboost.log_model(best_model, "best_model")

    print(f"Best model parameters: {best_params}")
    print(f"Test set MSE: {test_mse}")
    print(f"Test set MAE: {test_mae}")
    print(f"Test set MAPE: {test_mape}")

    # End the final run
    mlflow.end_run()
  return


def train_model(X_train, y_train, X_test, y_test):

  # Initialize TimeSeriesSplit
  tscv = TimeSeriesSplit(n_splits=4, test_size=100)

  # Initialize variables to keep track of the best model
  best_mse = float("inf")
  best_params = None
  best_model = None

  # Define parameter grid
  parameter_grid = {
      "max_depth": [3, 4, 5],  
      "learning_rate": [0.01, 0.05],
      "n_estimators": [100, 200, 300], 
      "colsample_bytree": [0.3, 0.5, 0.7],
      "lambda": [75, 100],  
      "alpha": [75, 100]
  }

  # Create all combinations of parameters
  param_combinations = list(product(
      parameter_grid["max_depth"], 
      parameter_grid["learning_rate"], 
      parameter_grid["n_estimators"], 
      parameter_grid["colsample_bytree"], 
      parameter_grid["lambda"],  # Explore for reducing overfitting - L2
      parameter_grid["alpha"]  # Used for high dimensionality - L1
  ))

  # Start the MLflow run
  for params in param_combinations:
      # Set model parameters
      param_dict = {
          "max_depth": params[0],
          "learning_rate": params[1],
          "n_estimators": params[2],
          "colsample_bytree": params[3],
          "lambda": params[4],
          "alpha": params[5]
      }

      pprint.PrettyPrinter(width=20).pprint(param_dict)

      with mlflow.start_run():
        # Log parameters
        for param, value in param_dict.items():
            mlflow.log_param(param, value)

        fold_mses = []
        fold_maes = []
        fold_mapes = []

        for i, (train_index, val_index) in enumerate(tscv.split(X_train)):
          print(f'Fold {i}')
          X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
          y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

          # Log fold number
          mlflow.log_param(f"fold_{i+1}_train_start", int(train_index[0]))
          mlflow.log_param(f"fold_{i+1}_train_end", int(train_index[-1]))
          mlflow.log_param(f"fold_{i+1}_val_start", int(val_index[0]))
          mlflow.log_param(f"fold_{i+1}_val_end", int(val_index[-1]))

          # Initialise XGBoost and train
          model = XGBRegressor(tree_method='hist', device = "cuda", **param_dict)
          model.fit(X_train_cv, y_train_cv)

          # Make predictions and evaluate model on Val
          y_val_pred = model.predict(X_val_cv)
          val_mae, val_mse, val_mape = evaluate_model(y_val_cv, y_val_pred)

          # Log metrics for each fold
          fold_maes.append(val_mae)
          fold_mses.append(val_mse)
          fold_mapes.append(val_mape)
          mlflow.log_metric(f"val_mae_fold_{i+1}", val_mae)
          mlflow.log_metric(f"val_mse_fold_{i+1}", val_mse)
          mlflow.log_metric(f"val_mape_fold_{i+1}", val_mape)

        # Log average MSE across all folds
        avg_mae = np.mean(fold_maes)
        avg_mse = np.mean(fold_mses)
        avg_mape = np.mean(fold_mapes)
        mlflow.log_metric("val_avg_mae", avg_mae)
        mlflow.log_metric("val_avg_mse", avg_mse)
        mlflow.log_metric("val_avg_mape", avg_mape)

        # Log the model for this run
        mlflow.xgboost.log_model(model, "model")

        # End the current run
        mlflow.end_run()

        # Check and log best model so far using MSE
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = param_dict
            best_model = model


  # retrain on full dataset using best params and log test metrics
  run_best_model(X_train, y_train, X_test, y_test, best_model, best_params)
        
  return


def run():
  # file paths
  CALLS_FP = '/content/data/case_data_calls.csv'
  RESERVATIONS_FP = '/content/data/case_data_reservations.csv'

  # window length
  WINDOW = 7
  PREDICTION_OFFSET = 6

  # download data
  repo_url = 'https://github.com/julianaddison/mlflow/'
  folder_path = 'data'  # Path to the folder within the repository
  download_github_folder(repo_url, folder_path)

  transformed_df = data_preparation(case_data_calls_fp=CALLS_FP,
                                    case_data_reservations_fp=RESERVATIONS_FP,
                                    window=WINDOW, 
                                    prediction_offset=PREDICTION_OFFSET)
  
  X_train, X_test, y_train, y_test = setup_train_test_data(calls_reservations_df=transformed_df,
                                                           window=WINDOW,
                                                           prediction_offset=PREDICTION_OFFSET)
  
  train_model(X_train, y_train, X_test, y_test)
  return


if __name__ == "__main__":
    run()
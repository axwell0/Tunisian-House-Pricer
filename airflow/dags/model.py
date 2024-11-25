import io
import logging
import os
import pickle

import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.mongo.hooks.mongo import MongoHook
from sklearn.model_selection import train_test_split

from utils import *

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 22, 16),  # Adjust the start date as needed
    'retries': 1,
}


# Function to download aggregated data from S3
def download_aggregated_data(**kwargs):
    s3_bucket = 'houses-transformed-data'
    prefix = ''
    s3_hook = S3Hook('aws_default')
    keys = s3_hook.list_keys(bucket_name=s3_bucket, prefix=prefix)
    if not keys:
        raise FileNotFoundError(f"No keys found in bucket '{s3_bucket}' with prefix '{prefix}'")
    folders = sorted(set(key.split('/')[0] for key in keys if '/' in key), reverse=True)
    if not folders:
        raise FileNotFoundError(f"No folders found in bucket '{s3_bucket}' with prefix '{prefix}'")
    newest_folder = folders[0]
    print(f"Newest folder: {newest_folder}")
    s3_key = f'{newest_folder}/aggregated/aggregated_data.csv'
    print(f"Fetching S3 key: {s3_key}")

    # Fetch the object
    obj = s3_hook.get_key(s3_key, bucket_name=s3_bucket)
    df = pd.read_csv(obj.get()['Body'])

    logging.info(f"Downloaded aggregate_df from s3://{s3_bucket}/{s3_key}")

    # Push DataFrame to XCom
    kwargs['ti'].xcom_push(key='aggregate_df', value=df.to_json(date_format='iso', orient='split'))


# Function to split data into train and test sets
def split_data(**kwargs):
    aggregate_df_json = kwargs['ti'].xcom_pull(key='aggregate_df')
    aggregate_df = pd.read_json(aggregate_df_json, orient='split')

    train_set, test_set = train_test_split(aggregate_df, test_size=0.2, random_state=42)

    kwargs['ti'].xcom_push(key='train_set', value=train_set.to_json(date_format='iso', orient='split'))
    kwargs['ti'].xcom_push(key='test_set', value=test_set.to_json(date_format='iso', orient='split'))

    logging.info("Data split into train and test sets.")


# Function to preprocess training data
def preprocess_training_data(**kwargs):
    train_set_json = kwargs['ti'].xcom_pull(key='train_set')
    train_set = pd.read_json(train_set_json, orient='split')

    train_set_preprocessed = preprocess_data(train_set)

    kwargs['ti'].xcom_push(key='train_set_preprocessed',
                           value=train_set_preprocessed.to_json(date_format='iso', orient='split'))

    logging.info("Preprocessed training data.")


# Function to preprocess testing data
def preprocess_testing_data(**kwargs):
    test_set_json = kwargs['ti'].xcom_pull(key='test_set')
    test_set = pd.read_json(test_set_json, orient='split')

    train_set_preprocessed_json = kwargs['ti'].xcom_pull(key='train_set_preprocessed')
    train_set_preprocessed = pd.read_json(train_set_preprocessed_json, orient='split')

    test_set_preprocessed = preprocss_test_data(
        test_set,
        train_set_preprocessed['city'],
        train_set_preprocessed['city_encoded'],
        train_set_preprocessed['state'],
        train_set_preprocessed['state_encoded']
    )

    kwargs['ti'].xcom_push(key='test_set_preprocessed',
                           value=test_set_preprocessed.to_json(date_format='iso', orient='split'))

    logging.info("Preprocessed testing data.")


def train_model(**kwargs):
    train_df_preprocessed_json = kwargs['ti'].xcom_pull(key='train_set_preprocessed')
    test_df_preprocessed_json = kwargs['ti'].xcom_pull(key='test_set_preprocessed')
    test_preprocessed = pd.read_json(test_df_preprocessed_json, orient='split')

    train_preprocessed = pd.read_json(train_df_preprocessed_json, orient='split')

    FEATURES = ['area', 'n_bedrooms', 'n_bathrooms', 'city_encoded', 'state_encoded', 'type_encoded']
    target = 'price'

    model_result = train_xgb_with_hyperopt(train_preprocessed, test_preprocessed, FEATURES, target)
    final_model = model_result.get('final_model')

    if not final_model:
        raise ValueError("Training function did not return 'final_model'.")

    logging.info("Model training completed.")

    # Serialize model using pickle
    model_pickle = pickle.dumps(final_model)

    logging.info("Model serialized using pickle.")

    # Prepare S3 details
    execution_date = kwargs.get('ds')  # 'YYYY-MM-DD' format
    date_str = datetime.strptime(execution_date, '%Y-%m-%d').strftime('%d-%m-%Y')
    models_bucket = 'finalmodels'
    model_s3_key = f"{date_str}/xgboost_model.pkl"  # Using .pkl extension for pickle files


    # Initialize S3Hook
    s3_hook = S3Hook('aws_default')


    train_preprocessed_csv = train_preprocessed.to_csv(index=False)
    train_preprocessed_s3_key = f"{date_str}/encodings.csv"
    s3_hook.load_string(
        string_data=train_preprocessed_csv,
        key=train_preprocessed_s3_key,
        bucket_name=models_bucket,
        replace=True
    )
    # Upload the pickled model to S3
    s3_hook.load_bytes(
        bytes_data=model_pickle,
        key=model_s3_key,
        bucket_name=models_bucket,
        replace=True
    )

    logging.info(f"Uploaded pickled model to s3://{models_bucket}/{model_s3_key}")




with DAG(
        dag_id='model',
        default_args=default_args,
        schedule_interval="0 12 * * 0",
        catchup=False,
        tags=['model'],
):
    start = EmptyOperator(task_id='start')
    download_data = PythonOperator(
        task_id='download_aggregated_data',
        python_callable=download_aggregated_data,
        provide_context=True,
    )
    split = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True,
    )
    preprocess_train = PythonOperator(
        task_id='preprocess_training_data',
        python_callable=preprocess_training_data,
        provide_context=True,
    )
    preprocess_test = PythonOperator(
        task_id='preprocess_testing_data',
        python_callable=preprocess_testing_data,
        provide_context=True,
    )
    train = PythonOperator(
        task_id='train_xgboost_model',
        python_callable=train_model,
        provide_context=True,
    )

    end = EmptyOperator(task_id='end')
    start >> download_data >> split >> [preprocess_train, preprocess_test] >> train >> end

import logging
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.mongo.hooks.mongo import MongoHook
from utils import get_collection
from transform_collections import transform_affare, transform_menzili,transform_aggregate



def connect_mongo():
    mongo_hook = MongoHook(conn_id="mongo_default")
    client = mongo_hook.get_conn()
    databases = client.list_database_names()
    print(f"Available databases: {databases}")
    return "database_processed_flag"


def apply_transform_affare(**kwargs):
    apply_transform_df(collection_name="Affare",
                       raw_s3_bucket='houses-raw-data',
                       transformed_s3_bucket='houses-transformed-data', transform_func=transform_affare, **kwargs)


def apply_transform_menzili(**kwargs):
    apply_transform_df(collection_name="menzili",
                       raw_s3_bucket='houses-raw-data',
                       transformed_s3_bucket='houses-transformed-data', transform_func=transform_menzili, **kwargs)


def apply_transform_df(collection_name: str, raw_s3_bucket: str, transformed_s3_bucket: str, transform_func, **kwargs):
    """
    Applies the transform_func function on collection_name data and uploads to transformed_s3_bucket.
    """
    s3_bucket_raw = raw_s3_bucket
    s3_bucket_transformed = transformed_s3_bucket
    date_str = datetime.strptime(kwargs['ds'], '%Y-%m-%d').strftime('%d-%m-%Y')

    s3_hook = S3Hook('aws_default')

    s3_key = kwargs['ti'].xcom_pull(key=f'{collection_name}_s3_key')
    df_obj = s3_hook.get_key(s3_key, bucket_name=s3_bucket_raw)
    df = pd.read_csv(df_obj.get()['Body'])
    transformed_df = transform_func(df)

    transformed_csv = transformed_df.to_csv(index=False)
    transformed_s3_key = f"{date_str}/{collection_name}/{collection_name}_transformed.csv"
    s3_hook.load_string(
        string_data=transformed_csv,
        key=transformed_s3_key,
        bucket_name=s3_bucket_transformed,
        replace=True
    )

    logging.info(f"Uploaded transformed {collection_name} data to s3://{s3_bucket_transformed}/{transformed_s3_key}")

    kwargs['ti'].xcom_push(key=f'{collection_name}_transformed_s3_key', value=transformed_s3_key)


def pull_data_from_source(*collection_names, **kwargs):
    """
    Pulls raw data from MongoDB collections and uploads to S3.
    """

    mongo_conn_id = 'mongo_default'
    s3_bucket_raw = 'houses-raw-data'
    date_str = datetime.strptime(kwargs['ds'], '%Y-%m-%d').strftime('%d-%m-%Y')

    mongo_hook = MongoHook(conn_id=mongo_conn_id)
    client = mongo_hook.get_conn()
    db = client["Houses"]

    s3_hook = S3Hook('aws_default')

    for collection_name in collection_names:
        df = get_collection(db.get_collection(collection_name))
        csv_buffer = df.to_csv(index=False)
        filename = f"{collection_name}.csv"
        s3_key = f"{date_str}/{collection_name}/{filename}"
        s3_hook.load_string(
            string_data=csv_buffer,
            key=s3_key,
            bucket_name=s3_bucket_raw,
            replace=True
        )

        logging.info(f"Uploaded {filename} to s3://{s3_bucket_raw}/{s3_key}")
        kwargs['ti'].xcom_push(key=f"{collection_name}_s3_key", value=s3_key)


def apply_transform_aggregate(**kwargs):
    """
    Applies the transform_aggregate function on both transformed Affare and menzili data and uploads the result to S3.
    """
    transformed_s3_bucket = 'houses-transformed-data'
    date_str = datetime.strptime(kwargs['ds'], '%Y-%m-%d').strftime('%d-%m-%Y')

    s3_hook = S3Hook('aws_default')
    affare_transformed_s3_key = kwargs['ti'].xcom_pull(key='Affare_transformed_s3_key')
    menzili_transformed_s3_key = kwargs['ti'].xcom_pull(key='menzili_transformed_s3_key')

    affare_obj = s3_hook.get_key(affare_transformed_s3_key, bucket_name=transformed_s3_bucket)
    affare_df = pd.read_csv(affare_obj.get()['Body'])

    menzili_obj = s3_hook.get_key(menzili_transformed_s3_key, bucket_name=transformed_s3_bucket)
    menzili_df = pd.read_csv(menzili_obj.get()['Body'])

    aggregated_df = transform_aggregate(affare_df, menzili_df)
    aggregated_csv = aggregated_df.to_csv(index=False)
    aggregated_s3_key = f"{date_str}/aggregated/aggregated_data.csv"
    s3_hook.load_string(
        string_data=aggregated_csv,
        key=aggregated_s3_key,
        bucket_name=transformed_s3_bucket,
        replace=True
    )

    logging.info(f"Uploaded aggregated data to s3://{transformed_s3_bucket}/{aggregated_s3_key}")
    kwargs['ti'].xcom_push(key='aggregated_data_s3_key', value=aggregated_s3_key)


with DAG(
        dag_id="preprocess_data",
        start_date=datetime(2024, 11, 24),
        schedule_interval=None,
        catchup=False

) as dag:
    connect_to_mongo_task = PythonOperator(
        task_id="connect_mongo",
        python_callable=connect_mongo,
    )
    pull_data = PythonOperator(
        task_id='pull_data_from_source',
        python_callable=pull_data_from_source,
        provide_context=True,
        op_args=['Affare', 'menzili'],
    )

    process_affare_task = PythonOperator(
        task_id='process_affare',
        python_callable=apply_transform_affare,
        provide_context=True,
    )
    process_menzili_task = PythonOperator(
        task_id='process_menzili',
        python_callable=apply_transform_menzili,
        provide_context=True,
    )
    process_aggregate_task = PythonOperator(
        task_id='process_aggregate',
        python_callable=apply_transform_aggregate,
        provide_context=True,
    )

    connect_to_mongo_task >> pull_data >> process_affare_task >> process_menzili_task >> process_aggregate_task

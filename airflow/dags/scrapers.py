from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 22, 16),
    'retries': 1,
}

with DAG(
        'House_scraper',
        default_args=default_args,
        description='Run Scrapy spider using Airflow and BashOperator',
        schedule_interval='0 9 * * *',  #
        catchup=False,
) as dag:
    start = EmptyOperator(task_id='start')
    scrape_affare = BashOperator(
        task_id='scrape_Affare',
        bash_command=(
            'cd /home/ubuntu/Affare && '
            'scrapy crawl Affare'
        )
    )
    scrape_menzili = BashOperator(
        task_id='scrape_Menzili',
        bash_command=(
            'cd /home/ubuntu/Affare && '
            'scrapy crawl menzili'
        )
    )
    trigger_preprocessing = TriggerDagRunOperator(
        task_id='trigger_preprocessing_dag',
        trigger_dag_id='preprocess_data',
        wait_for_completion=True,
        reset_dag_run=True, dag=dag
    )
    start >> [scrape_affare, scrape_menzili] >> trigger_preprocessing

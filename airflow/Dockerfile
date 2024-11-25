FROM apache/airflow:2.10.3


COPY ./Affare Affare
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER airflow

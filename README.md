We sought to build a house price prediction model for the tunisian market.

We developed scrapy spiders to scrape multiple tunisian websites and save them to a mongoDB database.

We then clean data, preprocess it, and train an XGBoost model.

Amazon S3 buckets are used to store data intermittently and store final models. Apache Airflow is used to orchestrate workflows (pre-processing, training and saving model...)

![image](https://github.com/user-attachments/assets/1b49af1f-5d9e-4494-ba70-64466fba80e5)

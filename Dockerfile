FROM apache/airflow:2.1.0
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

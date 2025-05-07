FROM apache/airflow:2.8.3
RUN pip install --no-cache-dir pandas sqlalchemy
RUN pip install --upgrade numpy setuptools
RUN pip install scikit-learn==0.24.2 --no-cache-dir

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
FROM quay.io/astronomer/astro-runtime:12.4.0-base

# Switch to root user to install packages
USER root

# Install necessary packages for HDFS and Spark integration
RUN apt-get update && apt-get install -y \
    krb5-multidev libkrb5-dev build-essential \
    libxml2-dev libxslt-dev zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the Docker image
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY ./utils /usr/local/airflow/utils
COPY ./configs /usr/local/airflow/configs

# Set environment variables for Airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////usr/local/airflow/airflow.db
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Set HDFS connection string
ENV AIRFLOW_CONN_WEBHDFS_DEFAULT=hdfs://ravikumar@hdfs-namenode:9870

# Install PySpark and HDFS dependencies if needed
RUN pip install pyspark hdfs

# Switch back to astro user
USER astro

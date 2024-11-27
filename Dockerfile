FROM quay.io/astronomer/astro-runtime:12.4.0-base

# Switch to root user to install packages
USER root

# Install necessary packages
RUN apt-get update && apt-get install -y krb5-multidev libkrb5-dev build-essential

# Copy requirements.txt into the Docker image
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY ./data_processing/data_pipelines/* /usr/local/airflow/data_pipelines/
COPY ./utils /usr/local/airflow/utils
COPY ./configs /usr/local/airflow/configs

# Switch back to astro user
USER astro

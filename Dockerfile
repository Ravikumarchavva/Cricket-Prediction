# Use Astro Runtime as the base image
FROM quay.io/astronomer/astro-runtime:12.4.0-base

# Switch to root user to install additional dependencies
USER root

# Install packages required for HDFS and Spark integration
RUN apt-get update && apt-get install -y \
    krb5-multidev libkrb5-dev build-essential \
    libxml2-dev libxslt-dev zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code to the image
COPY ./utils /usr/local/airflow/utils
COPY ./configs /usr/local/airflow/configs

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables for Airflow and HDFS
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////usr/local/airflow/airflow.db
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW_CONN_WEBHDFS_DEFAULT=hdfs://ravikumar@hdfs-namenode:9870

# Install PySpark and HDFS Python client
RUN pip install pyspark hdfs

# Switch back to astro user for running the application
USER astro

# Set the entrypoint for Airflow
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD ["webserver"]

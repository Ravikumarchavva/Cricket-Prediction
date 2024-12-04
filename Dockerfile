FROM quay.io/astronomer/astro-runtime:12.4.0-base

USER root

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk-headless \
    krb5-multidev libkrb5-dev build-essential \
    libxml2-dev libxslt-dev zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV HADOOP_USER_NAME=ravikumar

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./utils /usr/local/airflow/utils
COPY ./configs /usr/local/airflow/configs

USER ravikumar
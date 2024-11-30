FROM quay.io/astronomer/astro-runtime:12.4.0-base

USER root

RUN apt-get update && apt-get install -y \
    krb5-multidev libkrb5-dev build-essential \
    libxml2-dev libxslt-dev zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./utils /usr/local/airflow/utils
COPY ./configs /usr/local/airflow/configs

USER astro
# T20 Cricket Win Prediction Project

This project aims to predict the probability of a T20 cricket team winning at any given point in a match. By incorporating this model into a cricket app, we can attract more viewers by providing real-time predictions and insights during live matches. The solution uses a combination of **RNNs, CNNs, and DNNs** for analyzing sequential data (ball-by-ball), player statistics, and team-level statistics, respectively. These models are integrated into a deep neural network that yields real-time predictions. The final model is deployed as part of a portfolio site, with results showcased through visualizations and snapshots.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Modeling Approach](#ml-modeling-approach)
- [Results](#results)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

## Overview

The goal of this project is to provide **real-time predictions** on the likelihood of a team winning a T20 match, based on:

**Encoder**
- **Ball-by-Ball Sequences**: Utilizes an RNN to process the temporal sequence of game events.
- **Player Statistics**: Uses a CNN to capture player performance features.
- **Team Statistics**: Uses a DNN to aggregate overall team stats.
  
**Decoder**
  
These models are trained independently and then merged into a final DNN for comprehensive predictions.

## Directory Structure

Here's a high-level structure of the project folders:

```plaintext
.
├── Dockerfile
├── README.md
├── airflow.cfg
├── airflow_settings.yaml
├── apps
├── configs
│   ├── hp_config.yaml
│   └── spark_config.py
├── dags
│   └── t20.py
├── data_manuplation
│   ├── 1_preprocessing
│   ├── 2_eda
│   ├── 3_mergers
│   └── 4_filteringData
├── docker-compose.override.yml
├── include
│   ├── a_data_sources
│   ├── b_data_preprocessing
│   ├── c_data_merging
│   └── d_data_filtering
├── ml_modeling
│   ├── 1_data_preparation
│   ├── 2_naivetraining
│   ├── 3_augumented_training
│   ├── 4_hptuning
│   └── 5_selecting_best_model_to_onnx
├── public
├── requirements.txt
├── utils
└── ...
```

## Tech Stack

The project utilizes the following technologies:

- **Python**: Programming language for data processing and machine learning.
- **PyTorch**: Deep learning framework for building and training models.
- **Apache Spark**: Big data processing for data manipulation tasks.
- **Apache Airflow**: Workflow management for orchestrating data pipelines.
- **Hadoop HDFS**: Distributed file system for storing datasets.
- **Conda**: Environment management for package and dependency handling.
- **Jupyter Notebooks**: Interactive development environment for code and documentation.
- **Git**: Version control system for tracking changes in the codebase.
- **WandB**: Experiment Tracking, Hyperparameter tuning

## Data Processing Pipeline

The data processing pipeline consists of multiple stages and is orchestrated using **Apache Airflow**:

1. **Data Collection**: Pulls raw data from sources (e.g., Cricsheet) and ESPN cricket stats.
2. **Data Preprocessing**: Cleans and transforms raw data for player and team stats.
3. **Data Merging**: Combines individual datasets for player, team, and ball-by-ball statistics.
4. **Data Filtering**: Filters datasets for model training, focusing on relevant features to remove matches that are not in all 3 datasets (player, team, and ball-by-ball).

### Pipeline Diagram

The entire data pipeline is visualized in Airflow, with each step from data extraction to filtering organized as individual tasks.

![Airflow ETL Pipeline](./public/airflow_etl_pipeline.png)


## Ml modeling Approach

This solution employs a sophisticated **multi-model architecture**:

1. **RNN for Sequence**: Processes ball-by-ball data to capture temporal match dynamics.
2. **CNN for Player Stats**: Extracts features from player statistics, taking advantage of CNNs for feature aggregation.
3. **DNN for Team Stats**: Processes high-level team statistics for match conditions and overall team strength.
4. **Ensemble Model (DNN)**: Combines outputs from the RNN, CNN, and team DNN into a final DNN that predicts win probability.

*pytorchData is ignored due to large data size run \training\labeling\datasetpreparation.ipynb  file to get those files*

![Architecture Overview](./public/architectureOverview.png)

### Training Steps

1. Initial labeling and data preparation.
2. Naive training to test baseline model performance.
3. Evaluation and metrics analysis.
4. Hyperparameter tuning and additional evaluations.
5. Selecting best model from wandb sweep

### Results
The final model achieved an accuracy of 85% on the test set, which is tested across different overs

![Results](./public/results.png)

## Deployment

This model is deployed in a **portfolio website** as part of a static visual showcase rather than a live API to manage server costs. Users can view snapshots of model results and visualizations of prediction accuracy, as running real-time predictions requires constant data streaming, which would increase infrastructure costs.

## Getting Started

To run this project locally, follow the steps below:

### Prerequisites

1. Install the necessary dependencies using Conda:

    ```bash
    conda env create -f conda-env.yaml
    ```

2. Ensure **Apache Airflow**, **Spark**, **Hadoop**, **PyTorch**, and **WandB** are installed in your environment.

3. Set up **Airflow** and **HDFS** connections by updating the configuration in `./configs/spark_config.py`.

### Running the Pipeline

1. **Start Airflow**: Navigate to the project directory and run:

    ```bash
    astro dev start
    ```

2. **Trigger the Pipeline**: Access the Airflow webserver and start the `t20` DAG to process and transform cricket data.

### Training the Models

1. **Install PyTorch and WandB** if not already installed:

    ```bash
    pip install torch wandb
    ```

2. **Run Training Scripts**: Navigate to the `ml_modeling` directory and execute:

    ```bash
    python 2_naivetraining/train.py
    ```

3. **Hyperparameter Tuning**: Use WandB for experiment tracking and hyperparameter tuning.

### Deployment

Push the results to the portfolio website for visualization.

## Future Enhancements

1. **Live API**: Introduce a server that ingests real-time match data for live prediction capabilities.
2. **Advanced Model Tuning**: Explore ```ensemble techniques``` or enhace feature engineering using ```Domain Knowledge``` for enhanced predictions.
3. **Visual Dashboard**: Develop a dynamic dashboard to visualize ongoing predictions in a user-friendly format.

## Conclusion

This project showcases a comprehensive method for predicting the outcome of T20 cricket matches by utilizing an encoder-decoder architecture and integrating multiple machine learning models. By employing industry-standard tools like Hadoop HDFS, Apache Airflow, Apache Spark, PyTorch, and Weights & Biases (wandb), we've created a robust pipeline capable of handling large datasets and delivering real-time predictions. Incorporating this model into a cricket app can enhance user engagement by providing insightful analytics and live win probabilities.

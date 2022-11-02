"""Copyright 2022 by Artem Ustsov"""

import json
import logging
import os
import sys
from pathlib import Path

import click

from ml_project.data import read_data, write_data, split_train_val_data
from ml_project.data.load_data import download_data_from_s3
from ml_project.data.make_eda import make_eda_report
from ml_project.entities.train_pipeline_params import read_training_pipeline_params
from ml_project.features import make_features
from ml_project.features.build_features import extract_target, build_transformer, process_raw_data
from ml_project.models.process_model import serialize_model
from ml_project.models.fit_model import train_model
from ml_project.models.predict_model import predict_model
from ml_project.models.evaluate_model import evaluate_model

import mlflow

from ml_project.models.fit_model import create_inference_pipeline


if not os.path.isdir("logs"):
    os.mkdir("logs")

FORMAT_LOG = "%(asctime)s: %(message)s"
file_log = logging.FileHandler("logs/train_pipeline.log")
console_out = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    handlers=(file_log, console_out),
    format=FORMAT_LOG,
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    if training_pipeline_params.make_report:
        logger.info(f"Start making EDA report")
        make_eda_report(training_pipeline_params.input_data_path, training_pipeline_params.output_report_path)

    if training_pipeline_params.use_mlflow:
        logger.info(f"MLFlow Registry enable")
        mlflow.set_tracking_uri(training_pipeline_params.mlflow_uri)
        mlflow.create_experiment(training_pipeline_params.mlflow_experiment,
                                 artifact_location=training_pipeline_params.mlflow_artifact_location)
        # mlflow.set_experiment(training_pipeline_params.mlflow_experiment)

        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    downloading_params = training_pipeline_params.downloading_params
    if downloading_params:
        os.makedirs(downloading_params.output_folder, exist_ok=True)
        logger.info(f"Download data from {downloading_params.s3_bucket}")
        for path in downloading_params.paths:
            download_data_from_s3(
                downloading_params.s3_bucket,
                path,
                os.path.join(downloading_params.output_folder, Path(path).name),
            )

    logger.info(f"Start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    readable_data = process_raw_data(data, training_pipeline_params.feature_params)
    readable_data.to_csv(training_pipeline_params.output_clean_data_path, index_label=False)

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")


    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)
    logger.info(f"write train data")
    write_data(
        training_pipeline_params.output_proccessed_data_path,
        train_features,
    )

    logger.info(f"train_features.shape is {train_features.shape}")
    model = train_model(
        train_features, train_target,
        training_pipeline_params.train_params,
    )

    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(
        inference_pipeline,
        val_df,
    )
    metrics = evaluate_model(
        predicts,
        val_target,
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metrics is {metrics}")

    path_to_model = serialize_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
"""Copyright 2022 by Artem Ustsov"""

from typing import Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_proccessed_data_path: str
    output_model_path: str
    output_report_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    make_report: bool = False
    mlflow_uri: str = "http://10.0.0.8"
    mlflow_experiment: str = "demo"
    mlflow_artifact_location: str = "s3://ml_project/artifacts/"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        print(path)
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

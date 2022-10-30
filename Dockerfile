# Copyright 2022 by Artem Ustsov

FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz

COPY configs/ /configs
RUN mkdir -p /models

WORKDIR .

CMD ["ml_project_train", "configs/train_config.yaml"]

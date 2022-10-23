# VK Technopark-BMSTU | SEM II, ML OPS | HW_1

================================================================ 
  
Усцов Артем Алексеевич.  
Группа ML-21.  
Преподаватели: Михаил Марюфич


Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python ml_example/train_pipeline.py configs/train_config.yaml
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE            <- Licence desription (MIT default).
    ├── README.md          <- Homework description.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_project         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


DOCKER:
~~~
python setup.py sdist
docker build -t mikhailmar/train_made:v1 
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} mikhailmar/train_made:v1
~~~
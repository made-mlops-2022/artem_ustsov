"""Copyright 2022 by Artem Ustsov"""

import os
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi_health import health

from schemas import MedicalFeatures

app = FastAPI()

model = None
raw_transformer = None

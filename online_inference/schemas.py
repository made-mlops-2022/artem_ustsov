"""Copyright 2022 by Artem Ustsov"""

from typing import Literal

from pydantic import BaseModel


class MedicalFeatures(BaseModel):
    age: float
    sex: Literal[0, 1]
    chest_pain_type: Literal[0, 1, 2, 3]
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: Literal[0, 1]
    rest_ecg: Literal[0, 1, 2]
    max_heart_rate_achieved: float
    exercise_induced_angina: Literal[0, 1]
    oldpeak: float
    st_slope: Literal[0, 1, 2]
    st_depression: float
    num_major_vessels: float
    thalassemia: Literal[0, 1, 2]

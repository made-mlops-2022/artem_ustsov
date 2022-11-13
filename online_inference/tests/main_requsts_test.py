import json

import pytest
from fastapi.testclient import TestClient

from main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_predict_sick_endpoint():
    request = {
        "age": 0,
        "sex": 0,
        "chest_pain_type": 0,
        "resting_blood_pressure": 0,
        "cholesterol": 0,
        "fasting_blood_sugar": 0,
        "rest_ecg": 0,
        "max_heart_rate_achieved": 0,
        "exercise_induced_angina": 0,
        "st_depression": 0,
        "st_slope": 0,
        "num_major_vessels": 0,
        "thalassemia": 0
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'sick'}


def test_predict_healthy_endpoint():
    request = {
        "age": 0,
        "sex": 0,
        "chest_pain_type": 0,
        "resting_blood_pressure": 0,
        "cholesterol": 0,
        "fasting_blood_sugar": 0,
        "rest_ecg": 0,
        "max_heart_rate_achieved": 0,
        "exercise_induced_angina": 0,
        "st_depression": 0,
        "st_slope": 0,
        "num_major_vessels": 0,
        "thalassemia": 0
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'healthy'}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == 'Model is ready'


def test_missing_fields():
    request = {
        "age": 0,
        "sex": 0,
        "chest_pain_type": 0,
        "resting_blood_pressure": 0,
        "cholesterol": 0,
        "fasting_blood_sugar": 0,
        "rest_ecg": 0,
        "max_heart_rate_achieved": 0,
        "exercise_induced_angina": 0,
        "st_depression": 0,
        "st_slope": 0,
        "num_major_vessels": 0,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'


def test_categorical_fields():
    request = {
        "age": 0,
        "sex": 10,
        "chest_pain_type": 0,
        "resting_blood_pressure": 0,
        "cholesterol": 0,
        "fasting_blood_sugar": 0,
        "rest_ecg": 0,
        "max_heart_rate_achieved": 0,
        "exercise_induced_angina": 0,
        "st_depression": 0,
        "st_slope": 0,
        "num_major_vessels": 0,
        "thalassemia": 0
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'


def test_numerical_fields():
    request = {
        "age": 238,
        "sex": 10,
        "chest_pain_type": 0,
        "resting_blood_pressure": 0,
        "cholesterol": 0,
        "fasting_blood_sugar": 0,
        "rest_ecg": 0,
        "max_heart_rate_achieved": 0,
        "exercise_induced_angina": 0,
        "st_depression": 0,
        "st_slope": 0,
        "num_major_vessels": 0,
        "thalassemia": 0
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'wrong age value'
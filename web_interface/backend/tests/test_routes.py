from flask import Flask, jsonify
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_no_image(client):
    response = client.post('/predict')
    assert response.status_code == 400
    assert response.get_json() == {'error': 'No image uploaded'}

def test_predict_with_image(client):
    with open('tests/test_image.jpg', 'rb') as img:
        response = client.post('/predict', data={'image': img})
        assert response.status_code == 200
        assert isinstance(response.get_json(), list)  # Check if the response is a list of detections
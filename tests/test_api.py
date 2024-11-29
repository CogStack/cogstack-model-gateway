from fastapi.testclient import TestClient

from cogstack_model_gateway.gateway.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Enter the cult... I mean, the API."}

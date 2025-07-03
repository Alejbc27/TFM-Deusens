import pytest
from src.agents.api.main import app
from flask.testing import FlaskClient

client = app.test_client()

def test_regular_conversation():
    """
    Test a regular conversation with the agent.
    """
    response = client.post("/chat", json={
        "message": "Hola, ¿cómo estás?",
        "thread_id": "test-regular-conversation"
    })
    assert response.status_code == 200
    # Check the actual response structure from your API
    response_data = response.get_json()
    assert "response" in response_data or "message" in response_data

def test_hotel_info_in_rag():
    """
    Test asking for hotel info that is included in the RAG.
    """
    response = client.post("/chat", json={
        "message": "¿Cuál es la política de objetos perdidos?",
        "thread_id": "test-hotel-info-rag"
    })
    assert response.status_code == 200
    response_data = response.get_json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "90" in response_text.lower() and "30" in response_text.lower()

def test_hotel_info_not_in_rag():
    """
    Test asking for hotel info that is NOT included in the RAG.
    """
    response = client.post("/chat", json={
        "message": "¿Las habitaciones tienen minibar?",
        "thread_id": "test-hotel-info-not-rag"
    })
    assert response.status_code == 200
    response_data = response.get_json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "no" in response_text.lower() and "información" in response_text.lower()

def test_gym_availability_slot_available():
    """
    Test gym availability for a defined slot when it is available.
    """
    response = client.post("/chat", json={
        "message": "¿Hay disponibilidad en el gimnasio el lunes a las 10:00?",
        "thread_id": "test-gym-available"
    })
    assert response.status_code == 200
    response_data = response.get_json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "nombre" in response_text.lower() or "reservar" in response_text.lower()

def test_gym_availability_slot_not_available_accept_next():
    """
    Test gym availability for a defined slot when it is NOT available.
    """
    # First, ask for a slot that is not available
    response = client.post("/chat", json={
        "message": "¿Hay disponibilidad en el gimnasio el día 9 a las 8?",
        "thread_id": "test-gym-not-available"
    })
    assert response.status_code == 200
    response_data = response.get_json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "nombre" not in response_text.lower()
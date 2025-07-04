import requests
import pytest

API_URL = "http://agent-api:8081/chat"  # Use 'agent-api' for Docker Compose network

def test_regular_conversation():
    """
    Test a regular conversation with the agent.
    """
    payload = {
        "message": "Hola, ¿cómo estás?",
        "thread_id": "test-regular-conversation"
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    assert response.status_code == 200
    response_data = response.json()
    assert "response" in response_data or "message" in response_data

def test_hotel_info_in_rag():
    """
    Test asking for hotel info that is included in the RAG.
    """
    payload = {
        "message": "¿Cuál es la política de objetos perdidos?",
        "thread_id": "test-hotel-info-rag"
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "90" in response_text.lower() and "30" in response_text.lower()

def test_hotel_info_not_in_rag():
    """
    Test asking for hotel info that is NOT included in the RAG.
    """
    payload = {
        "message": "¿Las habitaciones tienen minibar?",
        "thread_id": "test-hotel-info-not-rag"
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "no" in response_text.lower() and "información" in response_text.lower()

def test_gym_availability_slot_available():
    """
    Test gym availability for a defined slot when it is available.
    """
    payload = {
        "message": "¿Hay disponibilidad en el gimnasio el lunes a las 10:00?",
        "thread_id": "test-gym-available"
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "nombre" in response_text.lower() or "reservar" in response_text.lower()

def test_gym_availability_slot_not_available_accept_next():
    """
    Test gym availability for a defined slot when it is NOT available.
    """
    payload = {
        "message": "¿Hay disponibilidad en el gimnasio el día 9 a las 8?",
        "thread_id": "test-gym-not-available"
    }
    response = requests.post(API_URL, json=payload, timeout=60)
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data.get("response", response_data.get("message", ""))
    assert "nombre" not in response_text.lower()
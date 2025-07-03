import pytest
from src.agents.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_regular_conversation():
    """
    Test a regular conversation with the agent.
    """
    response = client.post("/chat", json={"prompt": "Hola, ¿cómo estás?"})
    assert response.status_code == 200
    assert "hola" in response.json()["response"].lower() or "hotel" in response.json()["response"].lower()

def test_hotel_info_in_rag():
    """
    Test asking for hotel info that is included in the RAG.
    """
    response = client.post("/chat", json={"prompt": "¿Cuál es la política de objetos perdidos?"})
    assert response.status_code == 200
    assert "90" in response.json()["response"].lower() and "30" in response.json()["response"].lower()

def test_hotel_info_not_in_rag():
    """
    Test asking for hotel info that is NOT included in the RAG.
    """
    response = client.post("/chat", json={"prompt": "¿Las habitaciones tienen minibar?"})
    assert response.status_code == 200
    assert "no" in response.json()["response"].lower() and "información" in response.json()["response"].lower()

def test_gym_availability_slot_available():
    """
    Test gym availability for a defined slot when it is available.
    """
    response = client.post("/chat", json={"prompt": "¿Hay disponibilidad en el gimnasio el lunes a las 10:00?"})
    assert response.status_code == 200
    assert "nombre" in response.json()["response"].lower() or "reservar" in response.json()["response"].lower()

def test_gym_availability_slot_not_available_accept_next():
    """
    Test gym availability for a defined slot when it is NOT available.
    """
    # First, ask for a slot that is not available
    response = client.post("/chat", json={"prompt": "¿Hay disponibilidad en el gimnasio el día 9 a las 8?"})
    assert response.status_code == 200
    assert "nombre" not in response.json()["response"].lower()
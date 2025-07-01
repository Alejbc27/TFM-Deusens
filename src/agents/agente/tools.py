import requests
import json
import logging
from langchain_core.tools import tool
from config import RAG_SERVICE_URL, GYM_API_URL

logger = logging.getLogger(__name__)

@tool
def external_rag_search_tool(query: str, limit: int = 3, score_threshold: float = 0.3) -> str:
    """Busca en una base de conocimientos general información relacionada con la consulta del usuario."""
    logger.info(f"🛠️ Herramienta RAG Externa llamada con: query='{query}'")
    search_endpoint = f"{RAG_SERVICE_URL}/search"
    payload = {"query": query, "limit": limit, "score_threshold": score_threshold}
    try:
        response = requests.post(search_endpoint, json=payload, timeout=45)
        response.raise_for_status()
        search_data = response.json()
        if search_data and "results" in search_data and search_data["results"]:
            context_parts = [f"[Resultado {i+1}] Fuente: {doc.get('filename', 'N/A')}\nContenido: {doc.get('text', '')}" for i, doc in enumerate(search_data["results"])]
            return "Información recuperada:\n\n" + "\n---\n".join(context_parts)
        return "No se encontró información relevante."
    except requests.RequestException as e:
        return f"Error de red al contactar la base de conocimientos: {e}"
    except Exception as e:
        return f"Error inesperado en la herramienta RAG: {e}"

@tool
def check_gym_availability(target_date: str) -> str:
    """Comprueba la disponibilidad de plazas en el gimnasio para una fecha y hora (YYYY-MM-DDTHH:MM:SS)."""
    logger.info(f"🛠️ Herramienta Check Gym Availability llamada con: target_date='{target_date}'")
    url = f"{GYM_API_URL}/availability"
    payload = {"service_name": "gimnasio", "start_time": target_date}
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            slots_data = response.json()
            if isinstance(slots_data, list) and slots_data:
                start_times = [slot.get("start_time") for slot in slots_data if slot.get("start_time")][:5]
                if target_date in start_times:
                    return f"El horario {target_date} está disponible. Otros horarios cercanos: {json.dumps(start_times)}"
                return f"Horarios disponibles cerca de {target_date}: {json.dumps(start_times)}"
            return f"No hay horarios disponibles en el gimnasio para la fecha especificada ({target_date})."
        return f"No se pudo verificar la disponibilidad (código: {response.status_code})."
    except requests.RequestException as e:
        return f"Error de red al verificar disponibilidad: {e}"

@tool
def book_gym_slot(booking_date: str, user_name: str) -> str:
    """Reserva un horario (YYYY-MM-DDTHH:MM:SS) en el gimnasio para un usuario."""
    logger.info(f"🛠️ Herramienta Book Gym Slot llamada para {user_name} en {booking_date}.")
    avail_url = f"{GYM_API_URL}/availability"
    avail_payload = {"service_name": "gimnasio", "start_time": booking_date}
    headers = {"Content-Type": "application/json"}
    try:
        avail_response = requests.post(avail_url, json=avail_payload, headers=headers, timeout=15)
        if avail_response.status_code == 200 and avail_response.json():
            slot_to_book = next((s for s in avail_response.json() if s.get("start_time") == booking_date), None)
            if not slot_to_book or not slot_to_book.get("slot_id"):
                return f"El horario deseado {booking_date} ya no está disponible. Por favor, verifica de nuevo."

            book_url = f"{GYM_API_URL}/booking"
            booking_payload = {"slot_id": slot_to_book["slot_id"], "guest_name": user_name}
            book_response = requests.post(book_url, json=booking_payload, headers=headers, timeout=15)

            if book_response.status_code == 201:
                return f"Reserva exitosa para {user_name} a las {booking_date}."
            elif book_response.status_code == 409:
                return f"Conflicto de reserva: El horario {booking_date} ya está reservado."
            else:
                return f"Fallo la reserva (código {book_response.status_code}): {book_response.text[:100]}"
        return f"No se pudo confirmar la disponibilidad antes de reservar."
    except requests.RequestException as e:
        return f"Error de red al intentar reservar: {e}"

ALL_TOOLS_LIST = [external_rag_search_tool, check_gym_availability, book_gym_slot]

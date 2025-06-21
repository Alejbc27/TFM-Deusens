from datetime import date, datetime, time

from flask import Flask, request, jsonify, send_from_directory
from pydantic import BaseModel, ValidationError
from sqlalchemy import func
from flask_swagger_ui import get_swaggerui_blueprint
import os

from generator.main import get_session, Service, Slot, Booking


app = Flask(__name__)

@app.route("/docs/openapi.yml")
def openapi_spec():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    return send_from_directory(docs_dir, "openapi.yml")

# 2) Configuración de Swagger UI
SWAGGER_URL = "/docs"         # URL donde estará UI
API_URL = "/docs/openapi.yml" # Ruta donde tu app sirve el spec

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={                     # Opciones de configuración
        "app_name": "API Servicios Gimnasio & Sauna"
    },
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


class BookingCreate(BaseModel):
    slot_id: int
    guest_name: str


@app.route("/availability", methods=["POST"])
def get_service_availability():

    data = request.get_json(silent=True) or {}
    service_name = data.get("service_name")
    start_str = data.get("start_time")

    if service_name not in ["gimnasio", "sauna"]:
        return (
            jsonify({"error": "Servicio no encontrado. Use 'gimnasio' o 'sauna'."}),
            404,
        )
    if not start_str:
        return jsonify({"error": "El campo 'start_time' es obligatorio."}), 400

    session = get_session()
    try:
        if "T" in start_str:
            try:
                dt = datetime.fromisoformat(start_str)
            except ValueError:
                return (
                    jsonify(
                        {
                            "error": "Formato inválido para 'start_time'. Use YYYY-MM-DD o YYYY-MM-DDThh:mm:ss"
                        }
                    ),
                    400,
                )

            slot_with_count = (
                session.query(Slot, func.count(Booking.id).label("bookings_count"))
                .join(Service, Slot.service_id == Service.id)
                .outerjoin(Booking, Booking.slot_id == Slot.id)
                .filter(Service.name == service_name)
                .filter(Slot.start_time == dt)
                .group_by(Slot.id)
                .first()
            )
            if not slot_with_count:
                return jsonify([]), 200

            slot, cnt = slot_with_count
            available = slot.capacity - cnt
            return (
                jsonify(
                    [
                        {
                            "slot_id": slot.id,
                            "start_time": slot.start_time.isoformat(),
                            "total_capacity": slot.capacity,
                            "current_bookings": cnt,
                            "available_slots": available,
                        }
                    ]
                ),
                200,
            )

        try:
            day = date.fromisoformat(start_str)
        except ValueError:
            return (
                jsonify(
                    {
                        "error": "Formato inválido para 'start_time'. Use YYYY-MM-DD o YYYY-MM-DDThh:mm:ss"
                    }
                ),
                400,
            )

        start_of_day = datetime.combine(day, time(hour=8))
        end_of_day = datetime.combine(day, time(hour=21))

        results = (
            session.query(Slot, func.count(Booking.id).label("bookings_count"))
            .join(Service, Slot.service_id == Service.id)
            .outerjoin(Booking, Booking.slot_id == Slot.id)
            .filter(Service.name == service_name)
            .filter(Slot.start_time >= start_of_day)
            .filter(Slot.start_time < end_of_day)
            .group_by(Slot.id)
            .order_by(Slot.start_time)
            .limit(3)
            .all()
        )

        availability = []
        for slot, cnt in results:
            availability.append(
                {
                    "slot_id": slot.id,
                    "start_time": slot.start_time.isoformat(),
                    "total_capacity": slot.capacity,
                    "current_bookings": cnt,
                    "available_slots": slot.capacity - cnt,
                }
            )
        return jsonify(availability), 200

    finally:
        session.close()


@app.route("/booking", methods=["POST"])
def create_booking():

    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Cuerpo de la petición debe ser JSON."}), 400

    try:
        booking_data = BookingCreate.model_validate(json_data)
    except ValidationError as e:
        return (
            jsonify({"error": "Datos de entrada inválidos", "details": e.errors()}),
            422,
        )

    session = get_session()
    try:
        slot_to_book = (
            session.query(Slot)
            .filter(Slot.id == booking_data.slot_id)
            .with_for_update()
            .first()
        )

        if not slot_to_book:
            return jsonify({"error": "La franja horaria (slot) no existe."}), 404

        current_bookings = (
            session.query(Booking).filter(Booking.slot_id == slot_to_book.id).count()
        )

        if current_bookings >= slot_to_book.capacity:
            return (
                jsonify({"error": "No hay huecos disponibles en esta franja horaria."}),
                409,
            )

        new_booking = Booking(
            slot_id=booking_data.slot_id, guest_name=booking_data.guest_name
        )

        session.add(new_booking)
        session.commit()

        response_data = {
            "id": new_booking.id,
            "slot_id": new_booking.slot_id,
            "guest_name": new_booking.guest_name,
        }
        return jsonify(response_data), 201

    except Exception as e:
        session.rollback()
        return (
            jsonify({"error": "Ha ocurrido un error interno.", "details": str(e)}),
            500,
        )
    finally:
        session.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

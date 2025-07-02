import os
import re
import logging
import time
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agents.modules.agent import RagAgent
from src.agents.modules.tools import ALL_TOOLS_LIST
from src.agents.modules.redis_checkpointer import RedisCheckpointer
from src.agents.modules.metriclogger import MetricLogger
from src.agents.modules.config import OLLAMA_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
agent_instance = None
redis_checkpointer = None
metric_logger = None


def clean_agent_response(content):
    """Limpia tags de pensamiento y formatea la respuesta del agente."""
    if isinstance(content, str):
        # Remover tags <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>\s*\n?", "", content, flags=re.DOTALL).strip()
        return cleaned
    return str(content) if content else "Sin contenido en la respuesta."


def validate_thread_id(thread_id):
    """Valida el formato del thread_id para evitar problemas de seguridad."""
    if not thread_id or not isinstance(thread_id, str):
        return False
    if len(thread_id) > 100:  # Límite razonable
        return False
    if not re.match(r'^[a-zA-Z0-9_-]+$', thread_id):
        return False
    return True


def log_execution_metric(metric_name: str, execution_time: float):
    """Helper function para logging de métricas de ejecución."""
    if metric_logger:
        try:
            timestamp = datetime.now(timezone.utc)
            metric_logger.log_metric(timestamp, OLLAMA_MODEL_NAME, metric_name, execution_time)
        except Exception as e:
            logger.debug(f"Error registrando métrica {metric_name}: {e}")


def initialize_agent():
    """
    Inicializa el RagAgent instance y el Redis checkpointer.
    Se llama antes de la primera request.
    """
    global agent_instance, redis_checkpointer, metric_logger
    if agent_instance is None:
        try:
            logger.info("🚀 Inicializando RagAgent...")
            agent_instance = RagAgent(tools=ALL_TOOLS_LIST)
            
            # Inicializar checkpointer independiente para operaciones de gestión
            redis_checkpointer = RedisCheckpointer()
            
            # Inicializar metric logger
            try:
                metric_logger = MetricLogger()
                logger.info("📊 MetricLogger inicializado correctamente")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando MetricLogger: {e}")
                metric_logger = None
            
            logger.info("✅ RagAgent y RedisCheckpointer inicializados correctamente.")
        except Exception as e:
            logger.critical(f"❌ Error crítico inicializando el agente: {e}", exc_info=True)
            raise RuntimeError("No se pudo inicializar el agente.") from e


@app.before_request
def before_request_func():
    """Inicializar agente antes de la primera request."""
    initialize_agent()


@app.route('/chat', methods=['POST'])
def chat_with_agent():
    """
    Endpoint principal para interacción con el agente.
    Maneja persistencia automática de conversaciones.
    """
    start_time = time.time()
    tools_used = set()
    
    if agent_instance is None:
        return jsonify({"error": "Agente no inicializado."}), 503

    # Validar request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "El campo 'message' es requerido."}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({"error": "El mensaje no puede estar vacío."}), 400
    
    if len(message) > 2000:
        return jsonify({"error": "Mensaje demasiado largo (máximo 2000 caracteres)."}), 400

    # Manejo del thread_id
    if 'thread_id' not in data or not data['thread_id']:
        thread_id = f'user-fab1an12-{int(time.time())}'
        logger.info(f"🆕 Generando nuevo thread_id: {thread_id}")
    else:
        thread_id = data['thread_id']
        logger.info(f"🔄 Usando thread_id existente: {thread_id}")
    
    if not validate_thread_id(thread_id):
        return jsonify({"error": "thread_id inválido. Solo se permiten caracteres alfanuméricos, - y _."}), 400

    logger.info(f"📬 Mensaje recibido para thread '{thread_id}': '{message[:100]}{'...' if len(message) > 100 else ''}'")

    # Configuración para LangGraph
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Verificar estado de conversación existente
        session_exists = False
        existing_message_count = 0
        
        if redis_checkpointer:
            try:
                session_info = redis_checkpointer.get_session_info(thread_id)
                if session_info:
                    existing_message_count = session_info.get('message_count', 0)
                    session_exists = True
                    logger.info(f"🔍 Conversación existente: {existing_message_count} mensajes previos")
                else:
                    logger.info(f"🔍 Nueva conversación para thread '{thread_id}'")
            except Exception as e:
                logger.warning(f"Error verificando estado en Redis: {e}")

        # Procesar mensaje con el agente
        input_for_graph = {"messages": [HumanMessage(content=message)]}
        logger.info(f"🚀 Invocando grafo con mensaje: '{message[:50]}...'")

        final_state = agent_instance.graph.invoke(input_for_graph, config=config)
        
        if final_state and final_state.get('messages'):
            messages_after = final_state['messages']
            human_count = sum(1 for m in messages_after if isinstance(m, HumanMessage))
            ai_count = sum(1 for m in messages_after if isinstance(m, AIMessage))
            tool_count = sum(1 for m in messages_after if isinstance(m, ToolMessage))
            total_messages = len(messages_after)
            
            logger.info(f"✅ Estado final: {total_messages} mensajes (H:{human_count}, AI:{ai_count}, T:{tool_count})")
            
            # Verificar lógica de continuidad
            expected_human_count = existing_message_count // 2 + 1 if session_exists else 1
            if human_count != expected_human_count:
                logger.warning(f"⚠️ INCONSISTENCIA: Esperábamos {expected_human_count} HumanMessages, encontramos {human_count}")
            
            final_agent_message = messages_after[-1]
            
            # Detectar herramientas usadas
            for msg in messages_after:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if isinstance(tool_call, dict) and 'name' in tool_call:
                            tool_name = tool_call['name']
                            if tool_name == 'external_rag_search_tool':
                                tools_used.add('rag')
                            elif tool_name == 'check_gym_availability':
                                tools_used.add('availability')
                            elif tool_name == 'book_gym_slot':
                                tools_used.add('booking')
        else:
            logger.error(f"❌ No se pudo obtener estado final para thread '{thread_id}'")
            return jsonify({"error": "No se pudo obtener la respuesta final del agente."}), 500

        # Procesar respuesta final
        if isinstance(final_agent_message, AIMessage):
            response_content = clean_agent_response(final_agent_message.content)
            
            if not response_content and hasattr(final_agent_message, 'tool_calls') and final_agent_message.tool_calls:
                response_content = "Procesando su solicitud..."
                
        elif isinstance(final_agent_message, ToolMessage):
            response_content = f"Error procesando herramienta: {final_agent_message.content[:200]}..."
            logger.warning(f"Conversación terminó en ToolMessage para {thread_id}")
            
        else:
            response_content = clean_agent_response(getattr(final_agent_message, 'content', 'Sin contenido disponible.'))

        if not response_content or response_content.strip() == "":
            response_content = "El agente procesó su consulta pero no generó una respuesta textual."
            logger.warning(f"Respuesta vacía para thread {thread_id}")

        # Registrar métricas de ejecución
        total_execution_time = time.time() - start_time
        
        if not tools_used:
            log_execution_metric("ejecucion_sin_tools", total_execution_time)
        else:
            if 'rag' in tools_used:
                log_execution_metric("ejecucion_con_rag", total_execution_time)
            if 'availability' in tools_used:
                log_execution_metric("ejecucion_con_availability", total_execution_time)
            if 'booking' in tools_used:
                log_execution_metric("ejecucion_con_booking", total_execution_time)

        # Verificación final de estado
        final_message_count = existing_message_count
        if redis_checkpointer:
            session_info_after = redis_checkpointer.get_session_info(thread_id)
            if session_info_after:
                final_message_count = session_info_after.get('message_count', 0)
                increment = final_message_count - existing_message_count
                logger.info(f"📊 RESUMEN: {existing_message_count} -> {final_message_count} mensajes (+{increment})")
                
                if increment < 2:
                    logger.warning(f"⚠️ PROBLEMA: Solo se añadieron {increment} mensajes, esperábamos al menos 2")
            else:
                logger.warning("📊 No se pudo obtener info final de sesión")

        logger.info(f"💬 Respuesta: '{response_content[:100]}{'...' if len(response_content) > 100 else ''}'")
        logger.info(f"📊 Tiempo: {total_execution_time:.3f}s, Herramientas: {tools_used}")

        return jsonify({
            "response": response_content,
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": round(total_execution_time, 3),
            "tools_used": list(tools_used),
            "conversation_length": total_messages,
            "session_existed": session_exists,
            "message_increment": final_message_count - existing_message_count        
        })

    except Exception as e:
        execution_time = time.time() - start_time
        log_execution_metric("ejecucion_error", execution_time)
        
        logger.error(f"❌ Error durante la interacción del agente para '{thread_id}': {e}", exc_info=True)
        return jsonify({
            "error": f"Error interno del servidor: {str(e)[:100]}",
            "thread_id": thread_id,
            "execution_time": round(execution_time, 3)
        }), 500


@app.route('/sessions/<thread_id>', methods=['GET'])
def get_session_info(thread_id):
    """
    Obtiene información de una sesión específica.
    """
    if not validate_thread_id(thread_id):
        return jsonify({"error": "thread_id inválido."}), 400
    
    if not redis_checkpointer:
        return jsonify({"error": "Redis no disponible."}), 503
    
    try:
        session_info = redis_checkpointer.get_session_info(thread_id)
        
        if session_info:
            return jsonify({
                "thread_id": thread_id,
                "session_info": session_info,
                "exists": True
            })
        else:
            return jsonify({
                "thread_id": thread_id,
                "exists": False,
                "message": "Sesión no encontrada o expirada."
            }), 404
            
    except Exception as e:
        logger.error(f"Error obteniendo info de sesión {thread_id}: {e}")
        return jsonify({"error": "Error obteniendo información de sesión."}), 500


@app.route('/sessions/<thread_id>/clear', methods=['DELETE'])
def clear_session(thread_id):
    """
    Limpia una sesión específica.
    """
    if not validate_thread_id(thread_id):
        return jsonify({"error": "thread_id inválido."}), 400
    
    if not redis_checkpointer:
        return jsonify({"error": "Redis no disponible."}), 503
    
    try:
        # Limpiar del Redis checkpointer
        cleared = redis_checkpointer.clear_session(thread_id)
        
        # También limpiar del grafo del agente
        if agent_instance:
            config = {"configurable": {"thread_id": thread_id}}
            try:
                agent_instance.graph.clear_state(config)
            except Exception as e:
                logger.warning(f"Error limpiando estado del grafo para {thread_id}: {e}")
        
        if cleared:
            logger.info(f"🗑️ Sesión '{thread_id}' limpiada correctamente")
            return jsonify({
                "message": f"Sesión '{thread_id}' limpiada correctamente.",
                "thread_id": thread_id,
                "cleared": True
            })
        else:
            return jsonify({
                "message": f"Sesión '{thread_id}' no encontrada o ya estaba limpia.",
                "thread_id": thread_id,
                "cleared": False
            }), 404
            
    except Exception as e:
        logger.error(f"Error limpiando sesión {thread_id}: {e}")
        return jsonify({"error": "Error limpiando sesión."}), 500


@app.route('/sessions', methods=['GET'])
def list_sessions():
    """
    Lista sesiones activas con información básica.
    """
    if not redis_checkpointer:
        return jsonify({"error": "Redis no disponible."}), 503
    
    try:
        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 100)  # Entre 1 y 100
        
        sessions = redis_checkpointer.list_active_sessions(limit=limit)
        
        # Formatear la respuesta
        formatted_sessions = []
        for session in sessions:
            formatted_sessions.append({
                "thread_id": session.get("thread_id"),
                "message_count": session.get("message_count", 0),
                "last_updated": session.get("saved_at"),
                "last_message_type": session.get("last_message_type"),
                "user_login": session.get("user_login", "fab1an12")
            })
        
        return jsonify({
            "sessions": formatted_sessions,
            "total_found": len(formatted_sessions),
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error listando sesiones: {e}")
        return jsonify({"error": "Error obteniendo lista de sesiones."}), 500


@app.route('/sessions/cleanup', methods=['POST'])
def cleanup_old_sessions():
    """
    Endpoint para limpiar sesiones específicas o testing.
    """
    if not redis_checkpointer:
        return jsonify({"error": "Redis no disponible."}), 503
    
    data = request.get_json() or {}
    threads_to_clean = data.get('thread_ids', [])
    
    if not isinstance(threads_to_clean, list):
        return jsonify({"error": "thread_ids debe ser una lista."}), 400
    
    cleaned_count = 0
    errors = []
    
    for thread_id in threads_to_clean:
        if not validate_thread_id(thread_id):
            errors.append(f"thread_id inválido: {thread_id}")
            continue
            
        try:
            if redis_checkpointer.clear_session(thread_id):
                cleaned_count += 1
        except Exception as e:
            errors.append(f"Error limpiando {thread_id}: {str(e)}")
    
    return jsonify({
        "cleaned_sessions": cleaned_count,
        "errors": errors,
        "total_requested": len(threads_to_clean)
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint extendido con información de Redis y Métricas.
    """
    health_info = {
        "status": "ok", 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": "fab1an12"
    }
    
    # Verificar Redis
    if redis_checkpointer:
        try:
            redis_checkpointer.redis_client.ping()
            health_info["redis"] = "connected"
        except Exception as e:
            health_info["redis"] = f"error: {str(e)[:50]}"
            health_info["status"] = "degraded"
    else:
        health_info["redis"] = "not_initialized"
        health_info["status"] = "degraded"
    
    # Verificar agente
    health_info["agent"] = "initialized" if agent_instance else "not_initialized"
    
    # Verificar métricas
    if metric_logger:
        try:
            health_info["metrics"] = "initialized"
        except Exception as e:
            health_info["metrics"] = f"error: {str(e)[:50]}"
            health_info["status"] = "degraded"
    else:
        health_info["metrics"] = "not_initialized"
    
    return jsonify(health_info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
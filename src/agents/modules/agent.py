import traceback
import uuid
import json
from datetime import datetime

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .config import OLLAMA_MODEL_NAME
from .state import AgentState, get_current_agent_scratchpad, update_state_after_llm, update_state_after_tool
from .prompt import RAG_SYSTEM_PROMPT
from .redis_checkpointer import RedisCheckpointer

import logging
logger = logging.getLogger(__name__)

class RagAgent:
    def __init__(self, tools: list, ollama_model_name: str = OLLAMA_MODEL_NAME):
        self._tools_map = {t.name: t for t in tools}
        if not tools: raise ValueError("RagAgent requiere al menos una herramienta.")

        try:
            tools_as_json_schema = [convert_to_openai_tool(tool) for tool in tools]
            self._llm = ChatOllama(
                model=ollama_model_name,
                temperature=0.05,
            ).bind(tools=tools_as_json_schema)
            logger.info(f"🤖 LLM del Agente ({ollama_model_name}) inicializado. Herramientas vinculadas: {[t.name for t in tools]}.")
        except Exception as e:
            logger.error(f"❌ ERROR inicializando LLM ({ollama_model_name}): {e}\n{traceback.format_exc()}")
            raise

        # ✅ SOLUCIÓN: Simplificar el grafo para evitar duplicación
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', self.agent_node)  # Nodo principal unificado
        workflow.add_node('tools', self.tools_node)

        workflow.set_entry_point('agent')
        
        # ✅ FLUJO SIMPLIFICADO
        workflow.add_conditional_edges(
            'agent',
            self.should_continue,
            {'continue': 'tools', 'end': END}
        )
        workflow.add_edge('tools', 'agent')  # Tools regresa al agent
        
        try:
            redis_checkpointer = RedisCheckpointer()
            self.graph = workflow.compile(checkpointer=redis_checkpointer)
            logger.info("✅ Grafo del agente compilado con RedisCheckpointer.")
        except Exception as e:
            logger.error(f"❌ Error inicializando RedisCheckpointer, usando MemorySaver como fallback: {e}")
            self.graph = workflow.compile(checkpointer=MemorySaver())
            logger.warning("⚠️ Usando MemorySaver como fallback - Las conversaciones no persistirán")

    def should_continue(self, state: AgentState) -> str:
        """Decide si continuar con tools o terminar."""
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        logger.debug(f"  [Router] Último mensaje: {type(last_message).__name__ if last_message else 'None'}")
        
        # Si es un mensaje de herramienta o del usuario, el agente debe responder
        if isinstance(last_message, (ToolMessage, HumanMessage)):
            logger.debug("  [Router] -> Necesita respuesta del agente")
            return 'continue'
        
        # Si es respuesta del agente con tool_calls, ejecutar herramientas
        if isinstance(last_message, AIMessage):
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.debug("  [Router] -> Ejecutar herramientas")
                return 'continue'
            else:
                logger.debug("  [Router] -> Fin (respuesta final)")
                return 'end'
        
        return 'end'

    def agent_node(self, state: AgentState) -> dict:
        """Nodo principal que combina LLM y lógica de estado."""
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        # Logging del estado actual
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        logger.info(f"🤖 [Agent] Estado: {len(messages)} mensajes (H:{human_count}, AI:{ai_count}, T:{tool_count})")
        
        # Si el último mensaje es ToolMessage, actualizar estado primero
        if isinstance(last_message, ToolMessage):
            logger.debug("🔄 Actualizando estado después de herramienta...")
            state = self.update_state_after_tool(state)

        # Preparar mensajes para el LLM
        agent_scratchpad_content = get_current_agent_scratchpad(state)
        current_messages_for_llm = []
        system_prompt_with_scratchpad = RAG_SYSTEM_PROMPT.replace("{{agent_scratchpad}}", agent_scratchpad_content)

        # Procesar mensajes
        has_system_message = False
        for m in messages:
            if isinstance(m, SystemMessage):
                current_messages_for_llm.append(SystemMessage(content=system_prompt_with_scratchpad))
                has_system_message = True
            else:
                current_messages_for_llm.append(m)
        
        if not has_system_message:
            current_messages_for_llm.insert(0, SystemMessage(content=system_prompt_with_scratchpad))

        # Llamar al LLM
        logger.info(f"🤖 Llamando al LLM con {len(current_messages_for_llm)} mensajes...")
        try:
            ai_response = self._llm.invoke(current_messages_for_llm)
            logger.info(f"🤖 Respuesta LLM: {str(ai_response.content)[:100]}...")
        except Exception as e:
            logger.error(f"❌ ERROR en LLM: {e}")
            ai_response = AIMessage(content=f"Error: {e}", tool_calls=[])

        if not hasattr(ai_response, 'tool_calls'):
            ai_response.tool_calls = []

        # Procesar respuesta JSON si es necesario (workaround para Qwen)
        if isinstance(ai_response.content, str) and ai_response.content.strip().startswith("{"):
            try:
                content_json = json.loads(ai_response.content)
                if isinstance(content_json, dict) and "tool" in content_json and "tool_input" in content_json:
                    tool_name = content_json["tool"]
                    tool_args = content_json["tool_input"]
                    if tool_name in self._tools_map:
                        ai_response.tool_calls = [{"name": tool_name, "args": tool_args, "id": f"tc_{uuid.uuid4().hex}"}]
                        ai_response.content = ""
                        logger.info(f"🔧 Workaround: Convertido JSON a tool_call: {tool_name}")
            except:
                pass

        # ✅ CAMBIO CLAVE: Actualizar estado y devolver solo los mensajes NUEVOS
        temp_state = {**state, 'messages': messages + [ai_response]}
        updated_state = update_state_after_llm(temp_state)

        # Obtener solo los mensajes que se añadieron después del LLM
        new_messages = updated_state['messages'][len(messages):]

        logger.info(f"🤖 [Agent] Devolviendo {len(new_messages)} mensajes nuevos")
        logger.debug(f"🤖 [Agent] Nuevos mensajes: {[type(m).__name__ for m in new_messages]}")

        # ✅ DEVOLVER SOLO LOS MENSAJES NUEVOS
        return {'messages': new_messages}

    def tools_node(self, state: AgentState) -> dict:
        """Nodo de herramientas simplificado."""
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        logger.info(f"🔧 [Tools] Procesando herramientas...")
        
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.error("🔧 No hay tool_calls válidos")
            return state  # Devolver estado sin cambios

        tool_messages = []
        for tool_call in last_message.tool_calls:
            if not isinstance(tool_call, dict):
                continue
                
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args')
            tool_call_id = tool_call.get('id', f"tc_{uuid.uuid4().hex}")

            if tool_name not in self._tools_map:
                result = f"Error: Herramienta '{tool_name}' no disponible"
            else:
                try:
                    result = self._tools_map[tool_name].invoke(tool_args)
                    logger.info(f"🔧 Herramienta '{tool_name}' ejecutada")
                except Exception as e:
                    result = f"Error ejecutando {tool_name}: {e}"
                    logger.error(f"🔧 Error en herramienta '{tool_name}': {e}")

            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name
            ))

        # Devolver estado actualizado con los tool messages
        new_messages = messages + tool_messages
        logger.debug(f"🔧 [Tools] Devolviendo estado con {len(new_messages)} mensajes")
        return {**state, 'messages': new_messages}

    def update_state_after_tool(self, state: AgentState) -> AgentState:
        """Wrapper para actualización de estado después de herramientas."""
        check_gym_availability = self._tools_map.get('check_gym_availability')
        book_gym_slot = self._tools_map.get('book_gym_slot')
        return update_state_after_tool(state, check_gym_availability, book_gym_slot)
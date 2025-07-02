import traceback
import uuid
import json
import re
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
    """
    Agente RAG conversacional con persistencia en Redis y capacidades de herramientas.
    """
    
    def __init__(self, tools: list, ollama_model_name: str = OLLAMA_MODEL_NAME):
        """
        Inicializa el RagAgent con herramientas y modelo LLM.
        
        Args:
            tools: Lista de herramientas disponibles para el agente
            ollama_model_name: Nombre del modelo Ollama a utilizar
        """
        self._tools_map = {t.name: t for t in tools}
        if not tools:
            raise ValueError("RagAgent requiere al menos una herramienta.")

        # Inicializar LLM con herramientas
        try:
            tools_as_json_schema = [convert_to_openai_tool(tool) for tool in tools]
            self._llm = ChatOllama(
                model=ollama_model_name,
                temperature=0.05,
            ).bind(tools=tools_as_json_schema)
            logger.info(f"🤖 LLM del Agente ({ollama_model_name}) inicializado. Herramientas: {[t.name for t in tools]}")
        except Exception as e:
            logger.error(f"❌ Error inicializando LLM ({ollama_model_name}): {e}")
            raise

        # Construir grafo del agente
        self._build_graph()

    def _build_graph(self):
        """Construye el grafo de estados del agente."""
        workflow = StateGraph(AgentState)
        
        # Añadir nodos
        workflow.add_node('agent', self.agent_node)
        workflow.add_node('tools', self.tools_node)

        # Configurar flujo
        workflow.set_entry_point('agent')
        workflow.add_conditional_edges(
            'agent',
            self.should_continue,
            {'continue': 'tools', 'end': END}
        )
        workflow.add_edge('tools', 'agent')
        
        # Inicializar checkpointer
        try:
            redis_checkpointer = RedisCheckpointer()
            self.graph = workflow.compile(checkpointer=redis_checkpointer)
            logger.info("✅ Grafo del agente compilado con RedisCheckpointer")
        except Exception as e:
            logger.error(f"❌ Error con RedisCheckpointer, usando MemorySaver: {e}")
            self.graph = workflow.compile(checkpointer=MemorySaver())
            logger.warning("⚠️ Usando MemorySaver - Las conversaciones no persistirán")

    def should_continue(self, state: AgentState) -> str:
        """
        Determina el siguiente paso en el flujo del agente.
        Incluye workaround para modelos que devuelven JSON en lugar de tool_calls.
        
        Returns:
            'continue' para ejecutar herramientas, 'end' para finalizar
        """
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        logger.debug("🔍 [Router] Analizando último mensaje...")
        
        if isinstance(last_message, (ToolMessage, HumanMessage)):
            logger.debug("🔍 [Router] -> Mensaje de herramienta/usuario, continuar con agente")
            return 'continue'
        
        if isinstance(last_message, AIMessage):
            # Verificar tool_calls nativos primero
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_name = last_message.tool_calls[0].get('name', 'N/A')
                if tool_name in self._tools_map:
                    logger.info(f"🔍 [Router] -> Tool call nativo detectado: {tool_name}")
                    return 'continue'
                else:
                    logger.warning(f"🔍 [Router] -> Tool desconocido: {tool_name}, ignorando")
                    last_message.tool_calls = []
                    return 'end'
            
            # 🚨 WORKAROUND CRÍTICO: Verificar JSON content para Qwen
            if isinstance(last_message.content, str):
                # ✅ LIMPIAR CONTENIDO DE TAGS <think>...</think> ANTES DE VERIFICAR JSON
                cleaned_content = re.sub(r"<think>.*?</think>\s*\n?", "", last_message.content, flags=re.DOTALL).strip()
                logger.debug(f"🔍 [Router] Contenido limpio: '{cleaned_content[:100]}...'")
                
                if cleaned_content.startswith("{"):
                    logger.info("🔍 [Router] JSON detectado después de limpiar contenido")
                    try:
                        content_json = json.loads(cleaned_content)
                        logger.debug(f"🔍 [Router] JSON parseado: {content_json}")
                        
                        if (isinstance(content_json, dict) and 
                            "tool" in content_json and 
                            "tool_input" in content_json):
                            
                            tool_name = content_json["tool"]
                            tool_args = content_json["tool_input"]
                            
                            if tool_name in self._tools_map and isinstance(tool_args, dict):
                                logger.info(f"🔧 [Router WORKAROUND] JSON -> Tool call: {tool_name}")
                                
                                # Convertir JSON a tool_call nativo
                                last_message.tool_calls = [{
                                    "name": tool_name, 
                                    "args": tool_args, 
                                    "id": f"qwen_tc_{uuid.uuid4().hex}"
                                }]
                                last_message.content = ""  # Limpiar contenido JSON
                                
                                logger.info(f"🔧 [Router WORKAROUND] Tool call reconstruido para: {tool_name}")
                                return 'continue'
                            else:
                                logger.warning(f"🔧 [Router WORKAROUND] Tool desconocido o args inválidos: {tool_name}")
                        
                        elif isinstance(content_json, dict) and "answer" in content_json:
                            logger.info("🔍 [Router] JSON con 'answer' detectado, respuesta directa")
                            last_message.content = content_json["answer"]
                            last_message.tool_calls = []
                            return 'end'
                            
                    except json.JSONDecodeError:
                        logger.debug("🔍 [Router] Contenido limpio no es JSON válido")
                    except Exception as e:
                        logger.error(f"🔍 [Router WORKAROUND] Error procesando JSON: {e}")
            
            logger.info("🔍 [Router] -> Respuesta final sin herramientas")
            return 'end'
        
        logger.debug("🔍 [Router] -> Tipo de mensaje desconocido, finalizar")
        return 'end'

    def agent_node(self, state: AgentState) -> dict:
        """
        Nodo principal del agente que procesa mensajes con el LLM.
        
        Args:
            state: Estado actual del agente
            
        Returns:
            Dict con los mensajes nuevos generados
        """
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        # Log del estado actual
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        logger.info(f"🤖 [Agent] Estado: {len(messages)} mensajes (H:{human_count}, AI:{ai_count}, T:{tool_count})")
        
        # Actualizar estado si viene de herramientas
        if isinstance(last_message, ToolMessage):
            state = self.update_state_after_tool(state)

        # Preparar mensajes para el LLM
        agent_scratchpad_content = get_current_agent_scratchpad(state)
        current_messages_for_llm = self._prepare_messages_for_llm(messages, agent_scratchpad_content)

        # Invocar LLM
        logger.info(f"🤖 Llamando al LLM con {len(current_messages_for_llm)} mensajes...")
        try:
            ai_response = self._llm.invoke(current_messages_for_llm)
            logger.info(f"🤖 Respuesta LLM: {str(ai_response.content)[:100]}...")
        except Exception as e:
            logger.error(f"❌ Error en LLM: {e}")
            ai_response = AIMessage(content=f"Error: {e}", tool_calls=[])

        # Procesar respuesta y tool calls
        ai_response = self._process_ai_response(ai_response)

        # Actualizar estado y devolver solo mensajes nuevos
        temp_state = {**state, 'messages': messages + [ai_response]}
        updated_state = update_state_after_llm(temp_state)
        new_messages = updated_state['messages'][len(messages):]

        logger.info(f"🤖 [Agent] Devolviendo {len(new_messages)} mensajes nuevos")
        return {'messages': new_messages}

    def tools_node(self, state: AgentState) -> dict:
        """
        Nodo de herramientas que ejecuta las tool calls del agente.
        
        Args:
            state: Estado actual del agente
            
        Returns:
            Estado actualizado con los resultados de las herramientas
        """
        messages = state['messages']
        last_message = messages[-1] if messages else None
        
        logger.info("🔧 [Tools] Procesando herramientas...")
        
        if not self._has_valid_tool_calls(last_message):
            logger.error("🔧 No hay tool_calls válidos")
            return state

        # Ejecutar herramientas
        tool_messages = []
        for tool_call in last_message.tool_calls:
            if isinstance(tool_call, dict):
                tool_message = self._execute_tool_call(tool_call)
                tool_messages.append(tool_message)

        # Devolver estado actualizado
        new_messages = messages + tool_messages
        return {**state, 'messages': new_messages}

    def _prepare_messages_for_llm(self, messages, agent_scratchpad_content):
        """Prepara los mensajes para el LLM incluyendo el system prompt."""
        system_prompt_with_scratchpad = RAG_SYSTEM_PROMPT.replace(
            "{{agent_scratchpad}}", 
            agent_scratchpad_content
        )
        
        current_messages_for_llm = []
        has_system_message = False
        
        for message in messages:
            if isinstance(message, SystemMessage):
                current_messages_for_llm.append(SystemMessage(content=system_prompt_with_scratchpad))
                has_system_message = True
            else:
                current_messages_for_llm.append(message)
        
        if not has_system_message:
            current_messages_for_llm.insert(0, SystemMessage(content=system_prompt_with_scratchpad))
        
        return current_messages_for_llm

    def _process_ai_response(self, ai_response):
        """Procesa la respuesta del AI, incluyendo workaround para formato JSON."""
        if not hasattr(ai_response, 'tool_calls'):
            ai_response.tool_calls = []

        # Workaround para modelos que devuelven JSON en lugar de tool_calls
        if isinstance(ai_response.content, str) and ai_response.content.strip().startswith("{"):
            try:
                content_json = json.loads(ai_response.content)
                if self._is_valid_tool_json(content_json):
                    tool_name = content_json["tool"]
                    tool_args = content_json["tool_input"]
                    if tool_name in self._tools_map:
                        ai_response.tool_calls = [{
                            "name": tool_name, 
                            "args": tool_args, 
                            "id": f"tc_{uuid.uuid4().hex}"
                        }]
                        ai_response.content = ""
                        logger.info(f"🔧 Convertido JSON a tool_call: {tool_name}")
            except (json.JSONDecodeError, KeyError):
                pass

        return ai_response

    def _is_valid_tool_json(self, content_json):
        """Verifica si el JSON contiene una estructura de herramienta válida."""
        return (isinstance(content_json, dict) and 
                "tool" in content_json and 
                "tool_input" in content_json)

    def _has_valid_tool_calls(self, message):
        """Verifica si el mensaje tiene tool_calls válidos."""
        return (isinstance(message, AIMessage) and 
                hasattr(message, 'tool_calls') and 
                message.tool_calls)

    def _execute_tool_call(self, tool_call):
        """Ejecuta una herramienta específica y devuelve el ToolMessage."""
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tool_call_id = tool_call.get('id', f"tc_{uuid.uuid4().hex}")

        if tool_name not in self._tools_map:
            result = f"Error: Herramienta '{tool_name}' no disponible"
            logger.error(f"🔧 Herramienta no encontrada: {tool_name}")
        else:
            try:
                result = self._tools_map[tool_name].invoke(tool_args)
                logger.info(f"🔧 Herramienta '{tool_name}' ejecutada correctamente")
            except Exception as e:
                result = f"Error ejecutando {tool_name}: {e}"
                logger.error(f"🔧 Error en herramienta '{tool_name}': {e}")

        return ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        )

    def update_state_after_tool(self, state: AgentState) -> AgentState:
        """
        Actualiza el estado del agente después de ejecutar herramientas.
        
        Args:
            state: Estado actual del agente
            
        Returns:
            Estado actualizado
        """
        check_gym_availability = self._tools_map.get('check_gym_availability')
        book_gym_slot = self._tools_map.get('book_gym_slot')
        return update_state_after_tool(state, check_gym_availability, book_gym_slot)
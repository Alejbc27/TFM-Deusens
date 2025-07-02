import traceback
import uuid
import json
import re
from datetime import datetime

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
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
                temperature=0.05, # Un poco de temperatura para mejor conversación post-tool
                # format="json", # Eliminado para Qwen, confiando en el workaround y schema
            ).bind(tools=tools_as_json_schema)
            logger.info(f"🤖 LLM del Agente ({ollama_model_name}) inicializado. Herramientas vinculadas: {[t.name for t in tools]}.")
        except Exception as e:
            logger.error(f"❌ ERROR inicializando LLM ({ollama_model_name}): {e}\n{traceback.format_exc()}")
            raise

        # Definición del Grafo
        workflow = StateGraph(AgentState)
        workflow.add_node('call_llm', self.call_llm_node)
        workflow.add_node('invoke_tools_node', self.invoke_tools_node)
        # Nuevo nodo para actualizar el estado después de la respuesta del LLM (antes de decidir la herramienta)
        workflow.add_node('update_state_after_llm', self.update_state_after_llm)
        # Nuevo nodo para actualizar el estado después de la ejecución de la herramienta
        workflow.add_node('update_state_after_tool', self.update_state_after_tool)

        workflow.set_entry_point('call_llm')
        
        # Flujo: LLM -> update_state_after_llm -> Router -> (Tool o END)
        workflow.add_edge('call_llm', 'update_state_after_llm')
        workflow.add_conditional_edges(
            'update_state_after_llm', # El router ahora decide después de que el estado se haya actualizado con la última respuesta del LLM
            self.should_invoke_tool_router,
            {'invoke_tool': 'invoke_tools_node', 'respond_directly': END}
        )
        # Flujo: Tool -> update_state_after_tool -> LLM
        workflow.add_edge('invoke_tools_node', 'update_state_after_tool')
        workflow.add_edge('update_state_after_tool', 'call_llm')
        try:
            # Inicializar Redis Checkpointer
            redis_checkpointer = RedisCheckpointer()
            self.graph = workflow.compile(checkpointer=redis_checkpointer)
            logger.info("✅ Grafo del agente compilado con RedisCheckpointer.")
        except Exception as e:
            logger.error(f"❌ Error inicializando RedisCheckpointer, usando MemorySaver como fallback: {e}")
            self.graph = workflow.compile(checkpointer=MemorySaver())
            logger.warning("⚠️ Usando MemorySaver como fallback - Las conversaciones no persistirán")

    def update_state_after_llm(self, state: AgentState) -> AgentState:
        """Wrapper to call the state update function from the state module."""
        return update_state_after_llm(state)

    def update_state_after_tool(self, state: AgentState) -> AgentState:
        """Wrapper to call the state update function from the state module."""
        # Get tool references from the tools map
        check_gym_availability = self._tools_map.get('check_gym_availability')
        book_gym_slot = self._tools_map.get('book_gym_slot')
        return update_state_after_tool(state, check_gym_availability, book_gym_slot)

    def should_invoke_tool_router(self, state: AgentState) -> str:
        """
        Determina si se debe invocar una herramienta o responder directamente.
        Incluye workaround para modelos que devuelven JSON en lugar de tool_calls.
        ✅ VERSIÓN CORREGIDA - NO MODIFICA MENSAJES ORIGINALES
        """
        logger.debug("  [Router: Decidiendo siguiente paso...]")
        last_message = state['messages'][-1] if state['messages'] else None
        if not isinstance(last_message, AIMessage):
            logger.warning("    [Router] Último mensaje no es AIMessage.")
            return 'respond_directly'

        # ✅ VERIFICAR TOOL_CALLS NATIVOS PRIMERO
        if hasattr(last_message, 'tool_calls') and isinstance(last_message.tool_calls, list) and last_message.tool_calls:
            tool_name = last_message.tool_calls[0].get('name', 'N/A')
            if tool_name in self._tools_map:
                logger.info(f"    [Router] LLM solicitó herramienta vía .tool_calls: '{tool_name}'.")
                return 'invoke_tool'
            else:
                logger.warning(f"    [Router] LLM solicitó herramienta desconocida vía .tool_calls: '{tool_name}'. Ignorando.")
                return 'respond_directly'
        
        # ✅ WORKAROUND PARA QWEN - DETECTAR JSON SIN MODIFICAR EL MENSAJE ORIGINAL
        if isinstance(last_message.content, str):
            # Limpiar tags <think>...</think> si existen
            cleaned_content = re.sub(r"<think>.*?</think>\s*\n?", "", last_message.content, flags=re.DOTALL).strip()
            
            if cleaned_content.startswith("{"):
                try:
                    content_json = json.loads(cleaned_content)
                    logger.debug(f"    [Router] Contenido de AIMessage parseado como JSON: {content_json}")
                    
                    if isinstance(content_json, dict) and "tool" in content_json and "tool_input" in content_json:
                        tool_name = content_json["tool"]
                        tool_args = content_json["tool_input"]
                        
                        if tool_name in self._tools_map and isinstance(tool_args, dict):
                            logger.info(f"    [Router WORKAROUND] Detectada llamada a herramienta '{tool_name}' en .content.")
                            
                            # ✅ GUARDAR TOOL CALL EN STATE SIN MODIFICAR EL MENSAJE ORIGINAL
                            state['_pending_tool_call'] = {
                                "name": tool_name, 
                                "args": tool_args, 
                                "id": f"qwen_tc_{uuid.uuid4().hex}"
                            }
                            
                            logger.info(f"    [Router WORKAROUND] Tool call preparado: {tool_name}")
                            return 'invoke_tool'
                        else:
                            logger.warning(f"    [Router WORKAROUND] JSON en .content parece tool_call pero nombre ('{tool_name}') desconocido o args no dict.")
                            
                    elif isinstance(content_json, dict) and "answer" in content_json:
                        logger.info("    [Router] LLM devolvió JSON con 'answer', tratando como respuesta directa.")
                        return 'respond_directly'
                        
                except json.JSONDecodeError:
                    logger.debug(f"    [Router] Contenido no era JSON de tool_call esperado: '{last_message.content[:100]}...'")
                except Exception as e:
                    logger.error(f"    [Router WORKAROUND] Error inesperado procesando content_json: {e}\n{traceback.format_exc()}")

        logger.info(f"    [Router] LLM no solicitó herramienta. tool_calls: {getattr(last_message, 'tool_calls', 'N/A')}, content: '{str(last_message.content)[:100]}...'")
        return 'respond_directly'

    def call_llm_node(self, state: AgentState) -> dict:
        messages = state['messages']
        
        # Crear el system prompt, inyectando el scratchpad
        agent_scratchpad_content = get_current_agent_scratchpad(state)
        # logger.debug(f"    [LLM Node] Contenido del Scratchpad: {agent_scratchpad_content}")
        
        # Reemplazar el placeholder en el RAG_SYSTEM_PROMPT
        # O añadirlo como un mensaje separado. Añadirlo como mensaje es más limpio.
        current_messages_for_llm = []
        system_prompt_with_scratchpad = RAG_SYSTEM_PROMPT.replace("{{agent_scratchpad}}", agent_scratchpad_content)

        # Asegurar que el SystemMessage sea el primero y único
        # y que el scratchpad esté actualizado
        has_system_message = False
        for m in messages:
            if isinstance(m, SystemMessage):
                # Reemplazar el system message existente con el actualizado (con scratchpad)
                current_messages_for_llm.append(SystemMessage(content=system_prompt_with_scratchpad))
                has_system_message = True
            else:
                current_messages_for_llm.append(m)
        
        if not has_system_message:
            current_messages_for_llm.insert(0, SystemMessage(content=system_prompt_with_scratchpad))
            logger.info("    [LLM Node] Se antepuso RAG_SYSTEM_PROMPT con scratchpad.")
        else:
            logger.info("    [LLM Node] SystemMessage actualizado con scratchpad.")

        model_name_for_log = getattr(self._llm, 'model', getattr(getattr(self._llm, 'llm', None), 'model', 'modelo_desconocido'))
        logger.info(f"  [LLM Node] Llamando al LLM ({model_name_for_log}) con {len(current_messages_for_llm)} mensajes.")
        if current_messages_for_llm: logger.debug(f"    Último mensaje al LLM: {type(current_messages_for_llm[-1]).__name__} {str(current_messages_for_llm[-1].content)[:100]}...")
        
        try:
            ai_message_response = self._llm.invoke(current_messages_for_llm)
        except Exception as e:
            logger.error(f"❌ ERROR durante la invocación del LLM: {e}\n{traceback.format_exc()}")
            ai_message_response = AIMessage(content=f"Error al procesar con LLM: {e}", tool_calls=[])
        
        if not hasattr(ai_message_response, 'tool_calls') or not isinstance(ai_message_response.tool_calls, list):
            ai_message_response.tool_calls = [] 
        
        logger.info(f"    Respuesta CRUDA del LLM (AIMessage): tool_calls={ai_message_response.tool_calls}, content='{str(ai_message_response.content)[:200]}...'")
        
        # Devolver solo el nuevo AIMessage para ser añadido al estado.
        # Los nodos de actualización de estado se encargarán de modificar el estado existente.
        return {'messages': [ai_message_response]}

    def invoke_tools_node(self, state: AgentState) -> AgentState:
        """
        Nodo de invocación de herramientas.
        ✅ VERSIÓN CORREGIDA - MANEJA TOOL CALLS PENDIENTES DEL ROUTER
        """
        logger.info("  [Tools Node] Intentando invocar herramientas.")
        
        # ✅ VERIFICAR SI HAY TOOL CALL PENDIENTE DEL ROUTER
        if '_pending_tool_call' in state:
            logger.info("    [Tools Node] Procesando tool call pendiente del router.")
            tool_call = state['_pending_tool_call']
            
            # Ejecutar la herramienta pendiente
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args')
            tool_call_id = tool_call.get('id', f"tc_{uuid.uuid4().hex}")
            
            logger.info(f"    [Tools Node] Invocando herramienta: '{tool_name}' con args: {tool_args}")
            
            if tool_name not in self._tools_map:
                result_content = f"Error: Herramienta desconocida o no disponible: '{tool_name}'."
                logger.error(f"    [Tools Node] {result_content}")
            else:
                try:
                    # Validación de argumentos específica por herramienta
                    valid_args = True
                    missing_args = []
                    if tool_name == 'book_gym_slot':
                        if 'booking_date' not in tool_args: missing_args.append('booking_date')
                        if 'user_name' not in tool_args: missing_args.append('user_name')
                    elif tool_name == 'check_gym_availability' and 'target_date' not in tool_args:
                        missing_args.append('target_date')
                    elif tool_name == 'external_rag_search_tool' and 'query' not in tool_args:
                        missing_args.append('query')
                    
                    if missing_args:
                        result_content = f"Error: Para '{tool_name}' faltan los argumentos requeridos: {', '.join(missing_args)}. Argumentos recibidos: {tool_args}"
                        logger.warning(f"    [Tools Node] {result_content}")
                        valid_args = False
                        
                    if valid_args:
                        result_content = self._tools_map[tool_name].invoke(tool_args)
                        
                except Exception as e:
                    logger.error(f"      [Tools Node] ERROR ejecutando herramienta {tool_name}: {e}\n{traceback.format_exc()}")
                    result_content = f"Error al ejecutar la herramienta {tool_name}: {str(e)}"
            
            # Crear ToolMessage y limpiar el tool call pendiente
            tool_message = ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=result_content)
            new_state = {**state, "messages": state["messages"] + [tool_message]}
            
            # Limpiar el tool call pendiente
            if '_pending_tool_call' in new_state:
                del new_state['_pending_tool_call']
                
            logger.info(f"    [Tools Node] Herramienta '{tool_name}' invocada desde tool call pendiente.")
            return new_state
        
        # ✅ FLUJO ORIGINAL PARA TOOL_CALLS NATIVOS
        last_ai_message = state['messages'][-1] 

        if not (isinstance(last_ai_message, AIMessage) and hasattr(last_ai_message, 'tool_calls') and \
                isinstance(last_ai_message.tool_calls, list) and len(last_ai_message.tool_calls) > 0):
            error_msg = "Error: AIMessage inválida o sin tool_calls para invocación en invoke_tools_node."
            logger.error(f"    [Tools Node] {error_msg} tool_calls: {getattr(last_ai_message, 'tool_calls', 'NoAttr')}")
            return {**state, "messages": state["messages"] + [ToolMessage(content=error_msg, tool_call_id="error_no_valid_tool_calls")]}

        # Procesar tool_calls nativos
        tool_messages = []
        for tool_call in last_ai_message.tool_calls:
            if not isinstance(tool_call, dict):
                logger.error(f"    [Tools Node] Elemento tool_call no es un dict: {tool_call}")
                tool_messages.append(ToolMessage(content=f"Error: tool_call malformado: {tool_call}", tool_call_id=f"err_tc_{uuid.uuid4().hex}"))
                continue

            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args')
            tool_call_id = tool_call.get('id', f"tc_{uuid.uuid4().hex}") 

            if not tool_name or not isinstance(tool_args, dict):
                error_msg = f"Error: llamada a herramienta malformada. Nombre: '{tool_name}', Args: {type(tool_args)}"
                logger.error(f"    [Tools Node] {error_msg}")
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name or "unknown_tool"))
                continue
            
            logger.info(f"    [Tools Node] Invocando herramienta: '{tool_name}' con args: {tool_args}")
            if tool_name not in self._tools_map:
                result_content = f"Error: Herramienta desconocida o no disponible: '{tool_name}'."
                logger.error(f"    [Tools Node] {result_content}")
            else:
                try: 
                    valid_args = True; missing_args = []
                    if tool_name == 'book_gym_slot':
                        if 'booking_date' not in tool_args: missing_args.append('booking_date')
                        if 'user_name' not in tool_args: missing_args.append('user_name')
                    elif tool_name == 'check_gym_availability' and 'target_date' not in tool_args:
                        missing_args.append('target_date')
                    elif tool_name == 'external_rag_search_tool' and 'query' not in tool_args:
                        missing_args.append('query')
                    
                    if missing_args:
                        result_content = f"Error: Para '{tool_name}' faltan los argumentos requeridos: {', '.join(missing_args)}. Argumentos recibidos: {tool_args}"
                        logger.warning(f"    [Tools Node] {result_content}"); valid_args = False
                    if valid_args: result_content = self._tools_map[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.error(f"      [Tools Node] ERROR ejecutando herramienta {tool_name}: {e}\n{traceback.format_exc()}")
                    result_content = f"Error al ejecutar la herramienta {tool_name}: {str(e)}"
            
            tool_messages.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=result_content))
            logger.info(f"    [Tools Node] Herramienta '{tool_name}' invocada. Resultado añadido.")
        
        logger.info(f"    [Tools Node] Todos las tool_calls procesadas. {len(tool_messages)} ToolMessages generados.")
        return {**state, "messages": state["messages"] + tool_messages}
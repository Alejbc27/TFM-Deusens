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

        # Definición del Grafo - MANTENER LA ESTRUCTURA ORIGINAL
        workflow = StateGraph(AgentState)
        workflow.add_node('call_llm', self.call_llm_node)
        workflow.add_node('invoke_tools_node', self.invoke_tools_node)
        workflow.add_node('update_state_after_llm', self.update_state_after_llm)
        workflow.add_node('update_state_after_tool', self.update_state_after_tool)

        workflow.set_entry_point('call_llm')
        
        # Flujo original mantenido
        workflow.add_edge('call_llm', 'update_state_after_llm')
        workflow.add_conditional_edges(
            'update_state_after_llm',
            self.should_invoke_tool_router,
            {'invoke_tool': 'invoke_tools_node', 'respond_directly': END}
        )
        workflow.add_edge('invoke_tools_node', 'update_state_after_tool')
        workflow.add_edge('update_state_after_tool', 'call_llm')
        
        try:
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
        check_gym_availability = self._tools_map.get('check_gym_availability')
        book_gym_slot = self._tools_map.get('book_gym_slot')
        return update_state_after_tool(state, check_gym_availability, book_gym_slot)

    def should_invoke_tool_router(self, state: AgentState) -> str:
        logger.debug("  [Router: Decidiendo siguiente paso...]")
        last_message = state['messages'][-1] if state['messages'] else None
        if not isinstance(last_message, AIMessage):
            logger.warning("    [Router] Último mensaje no es AIMessage.")
            return 'respond_directly'

        if hasattr(last_message, 'tool_calls') and isinstance(last_message.tool_calls, list) and last_message.tool_calls:
            tool_name = last_message.tool_calls[0].get('name', 'N/A')
            if tool_name in self._tools_map:
                logger.info(f"    [Router] LLM solicitó herramienta vía .tool_calls: '{tool_name}'.")
                return 'invoke_tool'
            else:
                logger.warning(f"    [Router] LLM solicitó herramienta desconocida vía .tool_calls: '{tool_name}'. Ignorando.")
                last_message.tool_calls = []
                return 'respond_directly'
        
        if isinstance(last_message.content, str) and last_message.content.strip().startswith("{"):
            try:
                content_json = json.loads(last_message.content)
                logger.debug(f"    [Router] Contenido de AIMessage parseado como JSON: {content_json}")
                if isinstance(content_json, dict) and "tool" in content_json and "tool_input" in content_json:
                    tool_name = content_json["tool"]
                    tool_args = content_json["tool_input"]
                    if tool_name in self._tools_map and isinstance(tool_args, dict):
                        logger.warning(f"    [Router WORKAROUND] Detectada llamada a herramienta '{tool_name}' en .content.")
                        last_message.tool_calls = [{"name": tool_name, "args": tool_args, "id": f"qwen_tc_{uuid.uuid4().hex}"}]
                        last_message.content = "" 
                        logger.info(f"    [Router WORKAROUND] .tool_calls reconstruido: {last_message.tool_calls}")
                        return 'invoke_tool'
                    else:
                        logger.warning(f"    [Router WORKAROUND] JSON en .content parece tool_call pero nombre ('{tool_name}') desconocido o args no dict.")
                elif isinstance(content_json, dict) and "answer" in content_json:
                    logger.info("    [Router] LLM devolvió JSON con 'answer', tratando como respuesta directa.")
                    last_message.content = content_json["answer"]
                    last_message.tool_calls = []
                    return 'respond_directly'
            except json.JSONDecodeError:
                logger.debug(f"    [Router] Contenido no era JSON de tool_call esperado: '{last_message.content[:100]}...'")
            except Exception as e:
                logger.error(f"    [Router WORKAROUND] Error inesperado procesando content_json: {e}\n{traceback.format_exc()}")

        logger.info(f"    [Router] LLM no solicitó herramienta. tool_calls: {getattr(last_message, 'tool_calls', 'N/A')}, content: '{str(last_message.content)[:100]}...'")
        return 'respond_directly'

    def call_llm_node(self, state: AgentState) -> dict:
        messages = state['messages']
        
        # ✅ LOGGING MEJORADO: Contar tipos de mensajes
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        logger.info(f"  [LLM Node] Estado actual: {len(messages)} mensajes total (Human:{human_count}, AI:{ai_count}, Tool:{tool_count})")
        
        # Crear el system prompt, inyectando el scratchpad
        agent_scratchpad_content = get_current_agent_scratchpad(state)
        
        current_messages_for_llm = []
        system_prompt_with_scratchpad = RAG_SYSTEM_PROMPT.replace("{{agent_scratchpad}}", agent_scratchpad_content)

        # Asegurar que el SystemMessage sea el primero y único
        has_system_message = False
        for m in messages:
            if isinstance(m, SystemMessage):
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
        if current_messages_for_llm: 
            logger.debug(f"    Último mensaje al LLM: {type(current_messages_for_llm[-1]).__name__} {str(current_messages_for_llm[-1].content)[:100]}...")
        
        try:
            ai_message_response = self._llm.invoke(current_messages_for_llm)
        except Exception as e:
            logger.error(f"❌ ERROR durante la invocación del LLM: {e}\n{traceback.format_exc()}")
            ai_message_response = AIMessage(content=f"Error al procesar con LLM: {e}", tool_calls=[])
        
        if not hasattr(ai_message_response, 'tool_calls') or not isinstance(ai_message_response.tool_calls, list):
            ai_message_response.tool_calls = [] 
        
        logger.info(f"    Respuesta CRUDA del LLM (AIMessage): tool_calls={ai_message_response.tool_calls}, content='{str(ai_message_response.content)[:200]}...'")
        
        # ✅ MANTENER LA LÓGICA ORIGINAL - Solo devolver el nuevo mensaje
        logger.debug(f"  [LLM Node] Devolviendo 1 nuevo AIMessage")
        return {'messages': [ai_message_response]}

    def invoke_tools_node(self, state: AgentState) -> dict:  # ✅ CAMBIO: dict en lugar de AgentState
        logger.info("  [Tools Node] Intentando invocar herramientas.")
        
        # ✅ LOGGING MEJORADO
        messages = state['messages']
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        logger.info(f"  [Tools Node] Estado actual: {len(messages)} mensajes total (Human:{human_count}, AI:{ai_count}, Tool:{tool_count})")
        
        last_ai_message = messages[-1] 

        if not (isinstance(last_ai_message, AIMessage) and hasattr(last_ai_message, 'tool_calls') and \
                isinstance(last_ai_message.tool_calls, list) and len(last_ai_message.tool_calls) > 0):
            error_msg = "Error: AIMessage inválida o sin tool_calls para invocación en invoke_tools_node."
            logger.error(f"    [Tools Node] {error_msg} tool_calls: {getattr(last_ai_message, 'tool_calls', 'NoAttr')}")
            
            # ✅ CAMBIO: Devolver solo los mensajes nuevos
            error_tool_message = ToolMessage(content=error_msg, tool_call_id="error_no_valid_tool_calls")
            logger.debug(f"  [Tools Node] Devolviendo 1 ToolMessage de error")
            return {"messages": [error_tool_message]}

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
                    if valid_args: 
                        result_content = self._tools_map[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.error(f"      [Tools Node] ERROR ejecutando herramienta {tool_name}: {e}\n{traceback.format_exc()}")
                    result_content = f"Error al ejecutar la herramienta {tool_name}: {str(e)}"
            
            tool_messages.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=result_content))
            logger.info(f"    [Tools Node] Herramienta '{tool_name}' invocada. Resultado añadido.")
        
        logger.info(f"    [Tools Node] Todos las tool_calls procesadas. {len(tool_messages)} ToolMessages generados.")
        
        # ✅ CAMBIO PRINCIPAL: Solo devolver los nuevos ToolMessages
        logger.debug(f"  [Tools Node] Devolviendo {len(tool_messages)} ToolMessages nuevos")
        return {"messages": tool_messages}
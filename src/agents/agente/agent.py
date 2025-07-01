import operator
import logging
from typing import Annotated, TypedDict, Optional

from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.redis import RedisSaver

from config import OLLAMA_MODEL_NAME

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_name_for_gym_booking: Optional[str]
    gym_slot_iso_to_book: Optional[str]

RAG_SYSTEM_PROMPT = """Eres un asistente de servicio al cliente. Tu única función es responder preguntas usando herramientas. Eres eficiente, directo y no conversas.

**REGLAS FUNDAMENTALES:**
1.  **Analiza la pregunta del usuario.**
2.  **Decide si necesitas una herramienta para responder.**
3.  **Si necesitas una herramienta, LLÁMALA INMEDIATAMENTE.** Tu salida debe ser ÚNICAMENTE la llamada a la herramienta en formato JSON. NO escribas NADA MÁS. Sin explicaciones, sin saludos, sin '<think>'.
4.  **Si NO necesitas una herramienta, responde directamente.**
5.  **Después de usar una herramienta, usa su resultado para responder al usuario de forma concisa.**

---
**HERRAMIENTAS DISPONIBLES:**

1.  `external_rag_search_tool`:
    - **Cuándo usarla:** Para preguntas generales sobre el hotel, sus servicios (que no sean el gimnasio), políticas, horarios de check-in/out, etc.
    - **Ejemplo:** Usuario pregunta "¿Tienen piscina?". Llamas a `external_rag_search_tool(query="información sobre la piscina del hotel")`.

2.  `check_gym_availability`:
    - **Cuándo usarla:** Es el **PRIMER PASO OBLIGATORIO** para CUALQUIER consulta sobre el gimnasio. Úsala si el usuario pregunta por disponibilidad, horarios o quiere reservar.
    - **Ejemplo:** Usuario pregunta "Quiero reservar el gimnasio mañana por la mañana". Llamas a `check_gym_availability(target_date="YYYY-MM-DDT08:00:00")`.

3.  `book_gym_slot`:
    - **Cuándo usarla:** **SOLO Y EXCLUSIVAMENTE** si se cumplen TODAS las siguientes condiciones, que debes verificar en el contexto y el historial:
        a. Ya se ha llamado a `check_gym_availability` anteriormente.
        b. El usuario ha confirmado el horario EXACTO que quiere reservar (ej: "Sí, quiero las 10:00").
        c. Tienes el NOMBRE COMPLETO del usuario (ej: "Mi nombre es Carlos Portilla").
    - **NO la uses si falta información.** Si falta el nombre o la confirmación del horario, pide la información que falta.
    - **Ejemplo:** Si el contexto indica que el usuario es "Ana García" y ha confirmado "2025-07-15T11:00:00", llamas a `book_gym_slot(booking_date="2025-07-15T11:00:00", user_name="Ana García")`.

---
**CONTEXTO DE LA CONVERSACIÓN ACTUAL:**
Aquí tienes datos clave recordados de mensajes anteriores. Úsalos para tomar decisiones.
{{agent_scratchpad}}

---
**FLUJO DE TRABAJO PARA RESERVA DE GIMNASIO (SÍGUELO ESTRICTAMENTE):**

1.  **Input del Usuario:** "¿Hay sitio en el gym?"
    - **Tu Acción:** Llama a `check_gym_availability`.

2.  **Resultado de la Herramienta:** `check_gym_availability` devuelve horarios disponibles.
    - **Tu Acción:** Informa al usuario de los horarios. Pregunta cuál quiere y su nombre si no lo sabes. Ejemplo: "Hay disponibilidad a las 09:00, 10:00 y 11:00. ¿Cuál de estos horarios deseas reservar y cuál es tu nombre completo?"

3.  **Input del Usuario:** "Quiero a las 10:00. Me llamo Ana García."
    - **Tu Acción:** Ahora tienes toda la información. Llama a `book_gym_slot` con los datos confirmados.

4.  **Resultado de la Herramienta:** `book_gym_slot` devuelve un mensaje de éxito o fracaso.
    - **Tu Acción:** Informa al usuario del resultado final. Ejemplo: "Reserva confirmada para Ana García a las 10:00."

Recuerda: Si vas a usar una herramienta, no escribas nada más. Solo la llamada a la herramienta.

/nothink
"""

class RagAgent:
    def __init__(self, tools: list, checkpointer):
        self._tools_map = {t.name: t for t in tools}
        tools_as_json_schema = [convert_to_openai_tool(tool) for tool in tools]
        self._llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0).bind(tools=tools_as_json_schema)

        workflow = StateGraph(AgentState)
        workflow.add_node('call_llm', self.call_llm_node)
        workflow.add_node('invoke_tools_node', self.invoke_tools_node)
        workflow.add_node('update_state_from_user', self.update_state_from_user)

        workflow.set_entry_point('update_state_from_user')
        workflow.add_edge('update_state_from_user', 'call_llm')
        workflow.add_conditional_edges(
            'call_llm',
            self.should_invoke_tool_router,
            {'invoke_tool': 'invoke_tools_node', 'respond_directly': END}
        )
        workflow.add_edge('invoke_tools_node', 'call_llm')

        self.graph = workflow.compile(checkpointer=checkpointer)
        if isinstance(checkpointer, RedisSaver):
             logger.info("✅ Grafo del agente compilado con persistencia en Redis.")
        else:
             logger.info("✅ Grafo del agente compilado con persistencia en memoria (MemorySaver).")

    def _get_current_agent_scratchpad(self, state: AgentState) -> str:
        lines = []
        if state.get('gym_slot_iso_to_book'):
            lines.append(f"- Slot de gimnasio preseleccionado: {state['gym_slot_iso_to_book']}")
        if state.get('user_name_for_gym_booking'):
            lines.append(f"- Nombre de usuario recordado: {state['user_name_for_gym_booking']}")
        return "\n".join(lines) if lines else "No hay información de contexto."

    def update_state_from_user(self, state: AgentState) -> dict:
        last_human_message = state['messages'][-1].content
        if len(state['messages']) > 1:
            previous_message = state['messages'][-2]
            if isinstance(previous_message, AIMessage) and "nombre" in previous_message.content.lower():
                logger.info(f"Detectado posible nombre de usuario: '{last_human_message}'")
                state['user_name_for_gym_booking'] = last_human_message.strip()
        return {}

    def should_invoke_tool_router(self, state: AgentState) -> str:
        last_message = state['messages'][-1]
        return 'invoke_tool' if isinstance(last_message, AIMessage) and last_message.tool_calls else 'respond_directly'

    def call_llm_node(self, state: AgentState) -> dict:
        agent_scratchpad_content = self._get_current_agent_scratchpad(state)
        system_prompt_with_scratchpad = RAG_SYSTEM_PROMPT.replace("{{agent_scratchpad}}", agent_scratchpad_content)
        messages_for_llm = [SystemMessage(content=system_prompt_with_scratchpad)] + state['messages']

        logger.info(f"  [LLM Node] Llamando al LLM con {len(messages_for_llm)} mensajes (incl. System).")
        try:
            ai_message_response = self._llm.invoke(messages_for_llm)
            if ai_message_response.tool_calls:
                 logger.info(f"  [LLM Node] Modelo decidió usar herramienta.")
            return {'messages': [ai_message_response]}
        except Exception as e:
            logger.error(f"  [LLM Node] Error al llamar al LLM: {e}")
            error_msg = f"Error interno: no pude procesar tu solicitud con el modelo."
            return {'messages': [AIMessage(content=error_msg)]}

    def invoke_tools_node(self, state: AgentState) -> dict:
        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
             logger.warning("    [Tools Node] invoke_tools_node llamado sin tool_calls en el último mensaje.")
             return {"messages": [ToolMessage(content="Error interno: no se encontraron tool_calls válidas.", tool_call_id="error_tool_call")]}

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args')
            tool_call_id = tool_call.get('id')

            if tool_name not in self._tools_map:
                logger.warning(f"    [Tools Node] Herramienta desconocida solicitada: '{tool_name}'")
                result_content = f"Error: Herramienta '{tool_name}' no reconocida."
            else:
                logger.info(f"    [Tools Node] Invocando: '{tool_name}' con args: {tool_args}")
                try:
                    if not isinstance(tool_args, dict):
                         logger.warning(f"    [Tools Node] Argumentos inesperados para {tool_name}: {tool_args}. Intentando invocar.")
                         result_content = self._tools_map[tool_name].invoke(tool_args)
                    else:
                         result_content = self._tools_map[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.error(f"    [Tools Node] Error ejecutando la herramienta {tool_name} con args {tool_args}: {e}")
                    result_content = f"Error ejecutando la herramienta {tool_name}: {e}"
            tool_messages.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=str(result_content)))
        return {"messages": tool_messages}


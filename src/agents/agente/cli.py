import uuid
import re
import traceback
import logging
import requests

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config import logger, RAG_SERVICE_URL
from redis_setup import get_checkpointer
from tools import ALL_TOOLS_LIST
from agent import RagAgent

def main():
    logger.info("🚀 Iniciando el CLI del agente RAG...")

    redis_checkpointer, _ = get_checkpointer()

    if not redis_checkpointer:
        logger.error("Error crítico: No hay un checkpointer disponible. Saliendo.")
        exit(1)

    try:
        requests.get(f"{RAG_SERVICE_URL}/health", timeout=10).raise_for_status()
        logger.info("💚 Estado del servicio RAG: healthy")
    except requests.RequestException as e:
        logger.error(f"❌ No se pudo conectar al servicio RAG en {RAG_SERVICE_URL}: {e}.")

    try:
        rag_agent_instance = RagAgent(tools=ALL_TOOLS_LIST, checkpointer=redis_checkpointer)
        
        conversation_id = f"rag-docker-session-{uuid.uuid4().hex}"
        print(f"\n--- Iniciando conversación (ID: {conversation_id}) ---")
        print("💡 Escribe 'nueva' para empezar una conversación limpia, 'cargar [ID]' para cargar, o 'salir'.")

        while True:
            try:
                user_input = input("👤 Tú: ")
            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                break

            if user_input.lower() in ["salir", "exit", "quit"]:
                print("👋 ¡Adiós!")
                break

            if user_input.lower() == 'nueva':
                conversation_id = f"rag-docker-session-{uuid.uuid4().hex}"
                print(f"\n--- Empezando nueva conversación (ID: {conversation_id}) ---")
                continue
            
            if user_input.lower().startswith('cargar '):
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    potential_id = parts[1].strip()
                    config_to_check = {"configurable": {"thread_id": potential_id}}
                    try:
                        existing_state = rag_agent_instance.graph.get_state(config_to_check)
                        if existing_state:
                            conversation_id = potential_id
                            print(f"\n--- Cargada conversación existente (ID: {conversation_id}) ---")
                            if existing_state.values['messages']:
                                last_msg = existing_state.values['messages'][-1]
                                print(f"Último mensaje: {last_msg.pretty_repr()}")
                        else:
                             print(f"❗ ID de conversación '{potential_id}' encontrado pero está vacío. Continuando con él.")
                             conversation_id = potential_id
                    except (KeyError, IndexError, TypeError):
                        print(f"❗ ID de conversación '{potential_id}' no encontrado.")
                else:
                    print("❗ Uso: cargar [ID_de_conversacion]")
                continue

            if not user_input.strip():
                continue

            logger.info(f"📬 Usuario: '{user_input}' (Thread: {conversation_id})")
            config = {"configurable": {"thread_id": conversation_id}}
            input_for_graph = {"messages": [HumanMessage(content=user_input)]}
            final_ai_response_content = "El agente no generó una respuesta."

            try:
                final_state_values = rag_agent_instance.graph.invoke(input_for_graph, config=config)
                if final_state_values and final_state_values['messages']:
                    last_message_in_history = final_state_values['messages'][-1]
                    if isinstance(last_message_in_history, AIMessage) and not last_message_in_history.tool_calls:
                        final_content = str(last_message_in_history.content)
                        cleaned_content = re.sub(r"<think>.*?</think>\s*\n?", "", final_content, flags=re.DOTALL).strip()
                        final_ai_response_content = cleaned_content
                    elif isinstance(last_message_in_history, ToolMessage):
                        final_ai_response_content = "(Procesando resultados de la herramienta... Volveré a pensar)"
                    else:
                        final_ai_response_content = "(El agente está en un estado intermedio...)"

                print(f"🤖 Agente: {final_ai_response_content}")
                logger.info(f"💬 Agente (Output mostrado): '{final_ai_response_content}'")
                
                final_state = rag_agent_instance.graph.get_state(config)
                logger.info(f"  Estado persistido: user_name='{final_state.values.get('user_name_for_gym_booking')}', gym_slot='{final_state.values.get('gym_slot_iso_to_book')}'")

            except Exception as e:
                logger.critical(f"❌ Error durante la ejecución del grafo para thread {conversation_id}: {e}\n{traceback.format_exc()}")
                print("❗ Ocurrió un error durante la conversación. Consulta los logs.")
    except Exception as e:
        logger.critical(f"❌ Error crítico durante la inicialización o bucle principal: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()

import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple

from .config import REDIS_CONNECTION_POOL_CONFIG, REDIS_PREFIX, SESSION_TTL_SECONDS

logger = logging.getLogger(__name__)

class RedisCheckpointer(BaseCheckpointSaver):
    """
    Checkpointer personalizado que usa Redis para persistir el estado del agente.
    Mantiene compatibilidad completa con LangGraph mientras usa Redis como backend.
    """
    
    def __init__(self):
        """Inicializa el checkpointer con conexión Redis."""
        super().__init__()
        try:
            # Crear pool de conexiones Redis
            self.redis_pool = redis.ConnectionPool(**REDIS_CONNECTION_POOL_CONFIG)
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test de conexión
            self.redis_client.ping()
            logger.info("✅ RedisCheckpointer inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando RedisCheckpointer: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"No se pudo conectar a Redis: {e}") from e
    
    def _make_redis_key(self, thread_id: str, checkpoint_ns: str = "default") -> str:
        """Construye la key Redis para un thread específico."""
        # ✅ FIX: Asegurar que checkpoint_ns no sea vacío
        if not checkpoint_ns:
            checkpoint_ns = "default"
        return f"{REDIS_PREFIX}:{thread_id}:{checkpoint_ns}"
    
    def _make_metadata_key(self, thread_id: str, checkpoint_ns: str = "default") -> str:
        """Construye la key Redis para metadatos de un thread."""
        return f"{REDIS_PREFIX}:meta:{thread_id}:{checkpoint_ns}"
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> str:
        """Serializa un checkpoint a JSON para Redis."""
        try:
            # Convertir checkpoint a dict serializable
            checkpoint_dict = {
                "v": checkpoint.get("v", 1),
                "id": checkpoint.get("id"),
                "ts": checkpoint.get("ts"),
                "channel_values": checkpoint.get("channel_values", {}),
                "channel_versions": checkpoint.get("channel_versions", {}),
                "versions_seen": checkpoint.get("versions_seen", {}),
                "pending_sends": checkpoint.get("pending_sends", []),
            }
            
            # Serializar mensajes de forma especial
            if "channel_values" in checkpoint_dict and checkpoint_dict["channel_values"]:
                channel_values = checkpoint_dict["channel_values"]
                
                # Manejar los mensajes de LangChain
                if "messages" in channel_values:
                    messages = channel_values["messages"]
                    serialized_messages = []
                    
                    for msg in messages:
                        if hasattr(msg, 'dict'):
                            # LangChain message object
                            serialized_messages.append({
                                "type": msg.__class__.__name__,
                                "content": msg.content,
                                "additional_kwargs": getattr(msg, 'additional_kwargs', {}),
                                "tool_calls": getattr(msg, 'tool_calls', []),
                                "tool_call_id": getattr(msg, 'tool_call_id', None),
                                "name": getattr(msg, 'name', None),
                            })
                        else:
                            # Fallback para otros tipos
                            serialized_messages.append(str(msg))
                    
                    channel_values["messages"] = serialized_messages
            
            return json.dumps(checkpoint_dict, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error serializando checkpoint: {e}\n{traceback.format_exc()}")
            raise
    
    def _deserialize_checkpoint(self, data: str) -> Checkpoint:
        """Deserializa un checkpoint desde JSON."""
        try:
            checkpoint_dict = json.loads(data)
            
            # Reconstruir mensajes de LangChain
            if ("channel_values" in checkpoint_dict and 
                checkpoint_dict["channel_values"] and 
                "messages" in checkpoint_dict["channel_values"]):
                
                from langchain_core.messages import (
                    HumanMessage, AIMessage, SystemMessage, ToolMessage
                )
                
                messages_data = checkpoint_dict["channel_values"]["messages"]
                reconstructed_messages = []
                
                for msg_data in messages_data:
                    if isinstance(msg_data, dict) and "type" in msg_data:
                        msg_type = msg_data["type"]
                        content = msg_data.get("content", "")
                        
                        if msg_type == "HumanMessage":
                            reconstructed_messages.append(HumanMessage(content=content))
                        elif msg_type == "AIMessage":
                            ai_msg = AIMessage(
                                content=content,
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                            )
                            if msg_data.get("tool_calls"):
                                ai_msg.tool_calls = msg_data["tool_calls"]
                            reconstructed_messages.append(ai_msg)
                        elif msg_type == "SystemMessage":
                            reconstructed_messages.append(SystemMessage(content=content))
                        elif msg_type == "ToolMessage":
                            tool_msg = ToolMessage(
                                content=content,
                                tool_call_id=msg_data.get("tool_call_id", ""),
                            )
                            if msg_data.get("name"):
                                tool_msg.name = msg_data["name"]
                            reconstructed_messages.append(tool_msg)
                        else:
                            logger.warning(f"Tipo de mensaje desconocido: {msg_type}")
                            reconstructed_messages.append(HumanMessage(content=str(msg_data)))
                    else:
                        # Fallback para mensajes no estructurados
                        reconstructed_messages.append(HumanMessage(content=str(msg_data)))
                
                checkpoint_dict["channel_values"]["messages"] = reconstructed_messages
            
            return checkpoint_dict
            
        except Exception as e:
            logger.error(f"Error deserializando checkpoint: {e}\n{traceback.format_exc()}")
            raise
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")
            
            # ✅ FIX: Asegurar que checkpoint_ns nunca sea None o vacío
            if not checkpoint_ns:
                checkpoint_ns = "default"
                
            logger.debug(f"🔧 [GET] thread_id: {thread_id}, checkpoint_ns: {checkpoint_ns}")
            
            redis_key = self._make_redis_key(thread_id, checkpoint_ns)
            metadata_key = self._make_metadata_key(thread_id, checkpoint_ns)
            
            # Obtener checkpoint y metadata
            checkpoint_data = self.redis_client.get(redis_key)
            metadata_data = self.redis_client.get(metadata_key)
            
            if not checkpoint_data:
                logger.debug(f"No se encontró checkpoint para thread_id: {thread_id}")
                return None
            
            # Deserializar checkpoint
            checkpoint = self._deserialize_checkpoint(checkpoint_data)
            
            # Deserializar metadata si existe
            metadata = {}
            if metadata_data:
                try:
                    metadata = json.loads(metadata_data)
                except json.JSONDecodeError:
                    logger.warning(f"Metadata corrupta para {thread_id}, usando metadata vacía")
            
            # Crear CheckpointMetadata
            checkpoint_metadata = CheckpointMetadata(
                source=metadata.get("source", "update"),
                step=metadata.get("step", -1),
                writes=metadata.get("writes", {}),
                parents=metadata.get("parents", {}),
            )
            
            logger.debug(f"Checkpoint cargado para {thread_id}: {len(checkpoint.get('channel_values', {}).get('messages', []))} mensajes")
            
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=checkpoint_metadata,
                parent_config=None,  # Por simplicidad, no manejamos parent configs
            )
            
        except Exception as e:
            logger.error(f"Error obteniendo checkpoint: {e}\n{traceback.format_exc()}")
            return None
    
    def list_tuples(
        self, 
        config: RunnableConfig, 
        *, 
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> List[CheckpointTuple]:
        """
        Lista checkpoints para un thread. Por simplicidad, devolvemos solo el más reciente.
        """
        tuple_result = self.get_tuple(config)
        return [tuple_result] if tuple_result else []
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> RunnableConfig:
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")
            
            # ✅ FIX: Asegurar que checkpoint_ns nunca sea None o vacío
            if not checkpoint_ns:
                checkpoint_ns = "default"
            
            logger.debug(f"🔧 [PUT] thread_id: {thread_id}, checkpoint_ns: {checkpoint_ns}")
            
            redis_key = self._make_redis_key(thread_id, checkpoint_ns)
            metadata_key = self._make_metadata_key(thread_id, checkpoint_ns)
            
            # Serializar checkpoint
            checkpoint_json = self._serialize_checkpoint(checkpoint)
            
            # ✅ ARREGLAR: Manejo correcto del metadata (puede ser dict o objeto)
            if isinstance(metadata, dict):
                # Si metadata es un dict
                extended_metadata = {
                    "source": metadata.get("source", "update"),
                    "step": metadata.get("step", -1),
                    "writes": metadata.get("writes", {}),
                    "parents": metadata.get("parents", {}),
                }
            else:
                # Si metadata es un objeto CheckpointMetadata
                extended_metadata = {
                    "source": getattr(metadata, 'source', 'update'),
                    "step": getattr(metadata, 'step', -1),
                    "writes": getattr(metadata, 'writes', {}),
                    "parents": getattr(metadata, 'parents', {}),
                }
            
            # Añadir metadata adicional
            extended_metadata.update({
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "thread_id": thread_id,
                "user_login": "fab1an12",  # Usuario actual
            })
            
            # Agregar info de los mensajes para logging
            if (checkpoint.get("channel_values") and 
                "messages" in checkpoint["channel_values"]):
                message_count = len(checkpoint["channel_values"]["messages"])
                extended_metadata["message_count"] = message_count
                
                # Info del último mensaje
                if message_count > 0:
                    last_msg = checkpoint["channel_values"]["messages"][-1]
                    extended_metadata["last_message_type"] = type(last_msg).__name__
            
            metadata_json = json.dumps(extended_metadata, ensure_ascii=False, default=str)
            
            # Usar pipeline Redis para operaciones atómicas
            pipe = self.redis_client.pipeline()
            pipe.setex(redis_key, SESSION_TTL_SECONDS, checkpoint_json)
            pipe.setex(metadata_key, SESSION_TTL_SECONDS, metadata_json)
            pipe.execute()
            
            logger.info(f"✅ Checkpoint guardado para {thread_id}: {extended_metadata.get('message_count', 0)} mensajes")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Error guardando checkpoint: {e}\n{traceback.format_exc()}")
            raise
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Implementación requerida por BaseCheckpointSaver.
        Por simplicidad, no implementamos writes intermedias.
        """
        pass
    
    def clear_session(self, thread_id: str, checkpoint_ns: str = "default") -> bool:
        """
        Limpia una sesión específica de Redis.
        Método personalizado para gestión de sesiones.
        """
        try:
            redis_key = self._make_redis_key(thread_id, checkpoint_ns)
            metadata_key = self._make_metadata_key(thread_id, checkpoint_ns)
            
            deleted = self.redis_client.delete(redis_key, metadata_key)
            logger.info(f"🗑️ Sesión {thread_id} limpiada: {deleted} keys eliminadas")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Error limpiando sesión {thread_id}: {e}")
            return False
    
    def get_session_info(self, thread_id: str, checkpoint_ns: str = "default") -> Optional[Dict[str, Any]]:
        """
        Obtiene información de una sesión sin cargar el checkpoint completo.
        Método personalizado para gestión de sesiones.
        """
        try:
            metadata_key = self._make_metadata_key(thread_id, checkpoint_ns)
            metadata_data = self.redis_client.get(metadata_key)
            
            if metadata_data:
                return json.loads(metadata_data)
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo info de sesión {thread_id}: {e}")
            return None
    
    def list_active_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lista sesiones activas.
        Método personalizado para gestión de sesiones.
        """
        try:
            pattern = f"{REDIS_PREFIX}:meta:*"
            sessions = []
            
            for key in self.redis_client.scan_iter(match=pattern, count=limit):
                try:
                    # Extraer thread_id de la key
                    key_parts = key.split(":")
                    if len(key_parts) >= 3:
                        thread_id = key_parts[2]
                        
                        metadata_data = self.redis_client.get(key)
                        if metadata_data:
                            metadata = json.loads(metadata_data)
                            metadata["thread_id"] = thread_id
                            sessions.append(metadata)
                            
                except Exception as e:
                    logger.warning(f"Error procesando sesión {key}: {e}")
                    continue
            
            # Ordenar por fecha de guardado
            sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
            return sessions[:limit]
            
        except Exception as e:
            logger.error(f"Error listando sesiones activas: {e}")
            return []
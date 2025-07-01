import redis
import traceback
import logging
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import MemorySaver
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

logger = logging.getLogger(__name__)

def initialize_redis_checkpointer():
    """
    Inicializa la conexión a Redis y devuelve un checkpointer y un cliente.
    """
    try:
        redis_connection_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

        # 1. Crear cliente para checks de existencia
        client = redis.Redis.from_url(
            redis_connection_url,
            decode_responses=True
        )
        client.ping()
        logger.info("✅ Cliente Redis para verificaciones creado")

        # 2. Crear RedisSaver para LangGraph
        checkpointer = RedisSaver(redis_connection_url)
        checkpointer.setup()

        logger.info("✅ RedisSaver y cliente Redis inicializados")
        return checkpointer, client

    except redis.exceptions.AuthenticationError:
        logger.critical("❌ FALLO DE AUTENTICACIÓN")
        return None, None
    except Exception as e:
        logger.critical(f"❌ Error inesperado al conectar con Redis: {e}")
        return None, None

def get_checkpointer():
    """
    Obtiene el checkpointer de Redis o un fallback a memoria si falla la conexión.
    """
    checkpointer, client = initialize_redis_checkpointer()
    if checkpointer:
        return checkpointer, client
    else:
        logger.warning("⚠️ La conexión a Redis falló. Se usará MemorySaver como fallback (la conversación NO será persistente).")
        return MemorySaver(), None

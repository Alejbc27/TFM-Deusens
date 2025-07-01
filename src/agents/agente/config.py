import os
import logging

# --- Configuración de Servicios y Modelos ---
RAG_SERVICE_URL = os.getenv('RAG_SERVICE_URL', 'http://localhost:8080')
GYM_API_URL = os.getenv('GYM_API_URL', 'http://localhost:8000')
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', "caporti/qwen3-capor")

# --- Configuración de Conexión a Redis en DOCKER ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'redis_password')

# --- Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - AGENT - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

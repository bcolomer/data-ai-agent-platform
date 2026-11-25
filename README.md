# Data & AI Agent Platform

Arquitectura modular en Python orientada a la extracción estructurada de datos no estructurados, recuperación aumentada de contexto (RAG) y flujos agénticos con validación estricta de esquemas y control de consumo de tokens.

## Stack Técnico

- **Lenguaje:** Python 3.11+
- **LLM & Embeddings:** Google Gemini API (`google-genai`, `gemini-embedding-001`)
- **Base de Datos Vectorial:** ChromaDB (almacenamiento persistente local con distancia coseno HNSW)
- **Validación y Esquemas:** Pydantic v2 (`BaseModel`, `Field`, tipado estricto y manejo de nulos)
- **Gestión de Configuración:** `python-dotenv`
- **Control de Versiones:** Git / GitHub (flujo basado en ramas `feat/*` y Pull Requests)

## Estado del Proyecto (Roadmap)

- [x] **Fase 1: Extracción Estructurada y Resiliencia (`feat/structured-outputs`)**
  - Conexión con la API de Gemini y auditoría de consumo de tokens (`usage_metadata` y *thinking tokens*).
  - Definición de esquemas estrictos mediante Pydantic para transformar texto no estructurado en objetos tipados y JSON validado.
  - Manejo de valores ausentes (`T | None`) para evitar alucinaciones en campos opcionales.
  - Estrategia de *fallback* automático entre modelos ante errores transitorios de disponibilidad (`503 UNAVAILABLE`).
- [x] **Fase 2: Pipeline RAG (`feat/rag-pipeline`)**
  - Generación de embeddings de 3.072 dimensiones con `gemini-embedding-001`.
  - Indexación vectorial persistente en ChromaDB (`./chroma_db`) con ingesta idempotente (`count() == 0`) y búsqueda semántica por similitud coseno.
  - Recomendador de carrera estructurado (`rag_recommender.py`) con *grounding* estricto anti-alucinaciones, validación Pydantic, control de latencia y *fallback* automático entre modelos.
- [ ] **Fase 3: Orquestación Agéntica y API**
  - Flujos multi-paso con herramientas (*tool calling*) y exposición mediante FastAPI.

## Estructura Actual

```text
data-ai-agent-platform/
├── extractor_ofertas.py   # Extractor estructurado con validación Pydantic y fallback de modelos
├── vector_store.py        # Ingesta idempotente en ChromaDB y búsqueda semántica por embeddings
├── rag_recommender.py     # Pipeline RAG completo con grounding estricto y salida Pydantic
├── test_conexion.py       # Script de verificación de API y auditoría de tokens
├── test_embeddings.py     # Prueba unitaria de generación de vectores y similitud coseno
├── requirements.txt       # Dependencias del proyecto
└── .env.example           # Plantilla de variables de entorno requeridas
```

## Instalación y Uso Local

1. Clonar el repositorio y crear el entorno virtual:

```bash
git clone [https://github.com/bcolomer/data-ai-agent-platform.git](https://github.com/bcolomer/data-ai-agent-platform.git)
cd data-ai-agent-platform
python -m venv venv
source venv/Scripts/activate  # En Windows (Git Bash)
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar la clave de API en un archivo `.env` en la raíz:

```env
GEMINI_API_KEY="tu_clave_aqui"
```

4. Ejecutar los pipelines:

# Fase 1: Extracción estructurada desde texto libre
python extractor_ofertas.py

# Fase 2 (Paso 1): Indexar ofertas en la base de datos vectorial local (./chroma_db)
python vector_store.py

# Fase 2 (Paso 2): Ejecutar recomendación de carrera sobre el contexto recuperado
python rag_recommender.py
```
---



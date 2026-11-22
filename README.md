# data-ai-agent-platform
# Data & AI Agent Platform

Arquitectura modular en Python orientada a la extracción estructurada de datos no estructurados, recuperación aumentada de contexto (RAG) y flujos agénticos con validación estricta de esquemas y control de consumo de tokens.

## Stack Técnico

* **Lenguaje:** Python 3.11+
* **LLM & SDK:** Google Gemini API (`google-genai`)
* **Validación y Esquemas:** Pydantic v2 (`BaseModel`, `Field`, tipado estricto y manejo de nulos)
* **Gestión de Configuración:** `python-dotenv`
* **Control de Versiones:** Git / GitHub (flujo basado en ramas `feat/*` y Pull Requests)

## Estado del Proyecto (Roadmap)

- [x] **Fase 1: Extracción Estructurada y Resiliencia (`feat/structured-outputs`)**
  - Conexión con la API de Gemini y auditoría de consumo de tokens (`usage_metadata` y *thinking tokens*).
  - Definición de esquemas estrictos mediante Pydantic para transformar texto no estructurado en objetos tipados y JSON validado.
  - Manejo de valores ausentes (`T | None`) para evitar alucinaciones en campos opcionales.
  - Estrategia de *fallback* automático entre modelos ante errores transitorios de disponibilidad (`503 UNAVAILABLE`).
- [ ] **Fase 2: Pipeline RAG (Retrieval-Augmented Generation)**
  - Generación de embeddings (`gemini-embedding-001`), indexación vectorial y consulta contextual sobre documentos.
- [ ] **Fase 3: Orquestación Agéntica y API**
  - Flujos multi-paso con herramientas (*tool calling*) y exposición mediante FastAPI.

## Estructura Actual

```text
data-ai-agent-platform/
├── extractor_ofertas.py   # Extractor estructurado con validación Pydantic y fallback de modelos
├── test_conexion.py       # Script de verificación de API y auditoría de tokens
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

4. Ejecutar el extractor estructurado:

```bash
python extractor_ofertas.py
```
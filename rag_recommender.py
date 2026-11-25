import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
# Configuramos timeout de 15 segundos (15000 ms) para evitar esperas eternas si un servidor está saturado
client = genai.Client(http_options=types.HttpOptions(timeout=15_000))

# 1. Esquema Pydantic para la salida final del orientador
class RecomendacionCarrera(BaseModel):
    oferta_elegida: str = Field(description="Texto o resumen de la mejor oferta encontrada en el contexto")
    motivo_encaje: str = Field(description="Por qué esta oferta encaja con el perfil del candidato")
    requisitos_faltantes: list[str] = Field(description="Tecnologías o requisitos de la oferta que el candidato no mencionó")
    afinidad_estimada: int = Field(description="Puntuación de encaje del 1 al 100")

# 2. Conectamos a la base de datos vectorial ya creada en disco
chroma_client = chromadb.PersistentClient(path="./chroma_db")
coleccion = chroma_client.get_collection(name="ofertas_empleo")

def buscar_ofertas_relevantes(perfil_usuario: str, top_k: int = 2) -> list[str]:
    resp_emb = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[perfil_usuario],
    )
    vector_perfil = resp_emb.embeddings[0].values

    resultados = coleccion.query(
        query_embeddings=[vector_perfil],
        n_results=top_k,
    )
    return resultados["documents"][0]

def generar_recomendacion(perfil_usuario: str) -> RecomendacionCarrera:
# 1. Recuperamos los datos y medimos cuánto tarda ChromaDB
    t0_rag = time.perf_counter()
    # Paso R (Retrieval): Recuperamos las 2 ofertas más cercanas de ChromaDB
    ofertas_recuperadas = buscar_ofertas_relevantes(perfil_usuario, top_k=2)
    t_retrieval = time.perf_counter() - t0_rag
    print(f"[Retrieval ChromaDB] Completado en {t_retrieval:.2f}s\n")
    
    contexto_ofertas = "\n- ".join(ofertas_recuperadas)
   
    # Paso A (Augmented): Construimos el prompt pegando las ofertas recuperadas
    prompt = f"""
    Actúa como un orientador técnico de carrera.
    Analiza el perfil del candidato y selecciona la mejor oferta ÚNICAMENTE entre las siguientes opciones recuperadas de nuestra base de datos.
    No inventes ofertas ni requisitos que no aparezcan en el texto.

    PERFIL DEL CANDIDATO:
    {perfil_usuario}

    OFERTAS DISPONIBLES EN BASE DE DATOS:
    - {contexto_ofertas}
    """

# Paso G (Generation) con fallback automático por si un modelo da 503 o 404
    modelos_disponibles = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
    ]
    for modelo in modelos_disponibles:
        inicio = time.perf_counter()
        try:
            print(f"Consultando modelo: {modelo}...")
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RecomendacionCarrera,
                    temperature=0.2,
                ),
            )
            duracion = time.perf_counter() - inicio
            print(f"Respuesta recibida de {modelo} en {duracion:.2f}s")
            return respuesta.parsed
        except Exception as e:
            duracion = time.perf_counter() - inicio
            print(f"Fallo con {modelo} tras {duracion:.2f}s: {e}\nProbando siguiente modelo...\n")

    raise RuntimeError("Ningún modelo pudo completar la recomendación.")
            
if __name__ == "__main__":
    perfil = "Soy desarrolladora web con experiencia en backend, sé Python y SQL, y busco trabajo remoto construyendo APIs."
    print(f"Analizando perfil: '{perfil}'\n")

    recomendacion = generar_recomendacion(perfil)
    print("\n--- RECOMENDACIÓN ESTRUCTURADA ---")
    print(recomendacion.model_dump_json(indent=2))


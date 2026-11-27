import chromadb
from google.genai import types
from llm_client import generar_con_fallback
from vector_store import obtener_embeddings

# Conectamos a la base de datos vectorial existente en disco
chroma_client = chromadb.PersistentClient(path="./chroma_db")
coleccion = chroma_client.get_collection(name="ofertas_empleo")


# --- HERRAMIENTA 1: Búsqueda semántica en ChromaDB ---
def buscar_ofertas_db(perfil: str) -> str:
    """Busca ofertas de empleo relevantes en la base de datos vectorial ChromaDB según el perfil o tecnologías del candidato."""
    print(f"\n[TOOL CALL] Ejecutando 'buscar_ofertas_db' con perfil: '{perfil}'")
    vector_consulta = obtener_embeddings([perfil])[0]
    resultados = coleccion.query(
        query_embeddings=[vector_consulta],
        n_results=2,
    )
    documentos = resultados["documents"][0]
    if not documentos:
        return "No se encontraron ofertas relevantes en la base de datos."
    return "\n- ".join(documentos)


# --- HERRAMIENTA 2: Cálculo matemático determinista ---
def calcular_salario_neto_mensual(bruto_anual: float, pagas: int = 12) -> dict:
    """Calcula el salario neto mensual estimado en España a partir del salario bruto anual y el número de pagas (12 o 14)."""
    print(f"\n[TOOL CALL] Ejecutando 'calcular_salario_neto_mensual' con bruto_anual={bruto_anual}, pagas={pagas}")
    if bruto_anual <= 20000:
        retencion = 0.14
    elif bruto_anual <= 35000:
        retencion = 0.21
    elif bruto_anual <= 50000:
        retencion = 0.26
    else:
        retencion = 0.32

    neto_anual = bruto_anual * (1 - retencion)
    neto_mensual = round(neto_anual / pagas, 2)

    return {
        "bruto_anual": bruto_anual,
        "pagas": pagas,
        "retencion_estimada_pct": round(retencion * 100, 1),
        "neto_mensual_estimado": neto_mensual,
    }


# --- ORQUESTACIÓN DEL AGENTE ---
def ejecutar_agente(pregunta_usuario: str) -> str:
    configuracion_agente = types.GenerateContentConfig(
        system_instruction=(
            "Eres un asistente de orientación profesional y salarios en España. "
            "Usa tus herramientas para buscar ofertas reales en la base de datos o calcular salarios netos. "
            "Si el usuario pide datos que no puedes obtener con tus herramientas, dilo con honestidad y no inventes información."
        ),
        tools=[buscar_ofertas_db, calcular_salario_neto_mensual],
        temperature=0.2,
    )

    respuesta = generar_con_fallback(
        contenido=pregunta_usuario,
        config=configuracion_agente,
    )
    return respuesta.text


# --- PRUEBAS DE FUEGO (3 escenarios distintos) ---
if __name__ == "__main__":
    preguntas = [
        # Caso 1: Solo debe usar la herramienta de ChromaDB
        "Soy Data Engineer con experiencia en Python y SQL, ¿qué ofertas tenéis para mí?",
        # Caso 2: Solo debe usar la calculadora salarial
        "Si me ofrecen 38.000 euros brutos al año en 14 pagas, ¿cuánto cobraría limpio al mes?",
        # Caso 3: Debe encadenar AMBAS herramientas en una sola consulta
        "Búscame ofertas de desarrollo backend con Python y dime cuánto me quedaría neto al mes en 12 pagas si el puesto pagara 42.000 euros brutos.",
    ]

    for i, pregunta in enumerate(preguntas, start=1):
        print("=" * 60)
        print(f"PRUEBA {i} | USUARIO: {pregunta}")
        resultado = ejecutar_agente(pregunta)
        print(f"\nRESPUESTA AGENTE:\n{resultado}\n")

        #python agent_career.py
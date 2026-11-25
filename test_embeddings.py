import math
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def similitud_coseno(v1: list[float], v2: list[float]) -> float:
    producto_punto = sum(a * b for a, b in zip(v1, v2))
    magnitud_v1 = math.sqrt(sum(a * a for a in v1))
    magnitud_v2 = math.sqrt(sum(b * b for b in v2))
    return producto_punto / (magnitud_v1 * magnitud_v2)

documentos = [
    "Buscamos perfil técnico con dominio de Python, FastAPI y bases de datos SQL.",
    "Oferta para diseñador gráfico especialista en branding, Illustrator y Photoshop.",
    "Se requiere ingeniero de software backend para arquitectura de microservicios.",
]

consulta = "Busco trabajo programando servidores y APIs"

# 1. Generamos los embeddings en lote (batch)
resp_docs = client.models.embed_content(
    model="gemini-embedding-001",
    contents=documentos,
)
resp_query = client.models.embed_content(
    model="gemini-embedding-001",
    contents=consulta,
)

vector_consulta = resp_query.embeddings[0].values
print(f"Dimensión del vector: {len(vector_consulta)} floats")
print(f"Primeros 5 valores: {vector_consulta[:5]}\n")

# 2. Comparamos la consulta contra cada documento
resultados = []
for doc, emb in zip(documentos, resp_docs.embeddings):
    score = similitud_coseno(vector_consulta, emb.values)
    resultados.append((score, doc))

# Ordenamos de mayor a menor similitud
resultados.sort(reverse=True)

print(f"Consulta: '{consulta}'\n--- Ranking por similitud semántica ---")
for score, doc in resultados:
    print(f"[{score:.4f}] {doc}")
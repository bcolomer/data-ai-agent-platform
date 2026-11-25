import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

# 1. Inicializamos ChromaDB en modo persistente (guarda en disco en ./chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Creamos (o recuperamos si ya existe) una colección indicando que use similitud del coseno
coleccion = chroma_client.get_or_create_collection(
    name="ofertas_empleo",
    metadata={"hnsw:space": "cosine"},
)

#  Recibe lista en string y devuelve lista de vectores
def obtener_embeddings(textos: list[str]) -> list[list[float]]:
    respuesta = client.models.embed_content(
        model="gemini-embedding-001",
        contents=textos,
    )
    return [emb.values for emb in respuesta.embeddings]

# 2. Indexamos documentos solo si la colección está vacía (evita llamadas repetidas a la API)
if coleccion.count() == 0:
    print("Indexando ofertas en ChromaDB por primera vez...")
    ofertas = [
        "Se requiere ingeniero de software backend con Python, FastAPI y arquitectura de microservicios.",
        "Buscamos perfil fullstack junior con conocimientos en React, Node.js y bases de datos PostgreSQL.",
        "Oferta para diseñador gráfico especialista en branding, Illustrator y Photoshop.",
        "Ingeniero de Datos (Data Engineer) para pipelines ETL en Python, SQL y nube GCP.",
    ]
    ids = ["oferta_1", "oferta_2", "oferta_3", "oferta_4"]
    metadatos = [
        {"area": "backend", "modalidad": "remoto"},
        {"area": "fullstack", "modalidad": "hibrido"},
        {"area": "diseno", "modalidad": "presencial"},
        {"area": "data", "modalidad": "remoto"},
    ]

    vectores = obtener_embeddings(ofertas)

    coleccion.add(
        ids=ids,
        documents=ofertas,
        embeddings=vectores,
        metadatas=metadatos,
    )
    print(f"Guardadas {coleccion.count()} ofertas en disco.\n")
else:
    print(f"Cargadas {coleccion.count()} ofertas existentes desde disco (0 llamadas a API para indexar).\n")

# 3. Consulta semántica (Retrieval)
consulta = "Quiero trabajar desarrollando APIs y servicios en Python"
vector_consulta = obtener_embeddings([consulta])[0]

resultados = coleccion.query(
    query_embeddings=[vector_consulta],
    n_results=2,  # Solo nos devuelve los 2 resultados más cercanos
)

print(f"Consulta: '{consulta}'\n--- Top 2 documentos recuperados ---")
for doc_id, doc, meta, distancia in zip(
    resultados["ids"][0],
    resultados["documents"][0],
    resultados["metadatas"][0],
    resultados["distances"][0],
):
    # ChromaDB devuelve distancia coseno (0 = idéntico); similitud = 1 - distancia
    similitud = 1 - distancia
    print(f"[{similitud:.4f}] ({doc_id} | {meta['area']} - {meta['modalidad']}): {doc}")
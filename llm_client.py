import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"timeout": 15000},
)

MODELOS_POR_DEFECTO = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash", 
    "gemini-3.8-flash",
    "gemini-4-hydrogen"]


def generar_con_fallback(
    contenido: str,
    #Hace que esta función sirva igual para una llamada normal, una con 
    # JSON estricto (Pydantic) o una con herramientas (tools=[...]), como nuestro agente.
    config: types.GenerateContentConfig | None = None,
    modelos: list[str] = MODELOS_POR_DEFECTO,
):   
    """Ejecuta generación o sesión de agente con fallback automático entre modelos y log de latencia."""
    usa_herramientas = bool(config and config.tools)

    for modelo in modelos:
            try:
                inicio = time.perf_counter()
                if usa_herramientas:
                    chat = client.chats.create(model=modelo, config=config)
                    respuesta = chat.send_message(contenido)
                else:
                    respuesta = client.models.generate_content(
                        model=modelo,
                        contents=contenido,
                        config=config,
                    )
                latencia = time.perf_counter() - inicio
                print(f"[INFO] Modelo: {modelo} | Latencia: {latencia:.2f}s")
                return respuesta

            except Exception as e:
                print(f"[AVISO] Error con {modelo}: {e}. Probando siguiente modelo...")

    raise RuntimeError("Todos los modelos fallaron al procesar la solicitud.")
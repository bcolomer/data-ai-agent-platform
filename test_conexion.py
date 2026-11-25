import os
from dotenv import load_dotenv #Es una herramienta especializada en abrir archivos .env y cargar su contenido en la memoria temporal del programa.
from google import genai #Es el kit oficial de Google (el SDK) que sabe cómo hablar con los modelos Gemini por internet.

# 1. Cargamos la clave secreta desde el archivo .env
load_dotenv()

# 2. Inicializamos el cliente de Gemini (lee GEMINI_API_KEY automáticamente)
client = genai.Client()

# 3. Hacemos una primera llamada sencilla
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Salúdame en una frase corta como mi mentor de AI Engineering."
)


""" from google.genai import types  # Importamos las opciones de configuración

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Salúdame en una frase corta como mi mentor de AI Engineering.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
) """

print("\nRespuesta de Gemini:")
print(response.text)

print("\n--- Consumo de Tokens ---")
print("Tokens de entrada (tu prompt):", response.usage_metadata.prompt_token_count)
print("Tokens de salida (respuesta):", response.usage_metadata.candidates_token_count)
print("Total de tokens:", response.usage_metadata.total_token_count)


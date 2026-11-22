from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client()

class OfertaEmpleo(BaseModel):
    empresa: str = Field(description="Nombre de la empresa que publica la oferta")
    puesto: str = Field(description="Título exacto del puesto")
    tecnologias: list[str] = Field(description="Lista de herramientas, lenguajes o frameworks")
    remoto: bool | None = Field(
        default=None,
        description="True si es 100% remoto, False si es híbrido/presencial, null si no se menciona"
    )
    salario_anual: int | None = Field(
        default=None,
        description="Salario bruto anual en euros si aparece en el texto, null si no se indica"
    )

texto_oferta = """
En CloudPulse buscamos un Data & AI Engineer para nuestro equipo de arquitectura.
Imprescindible experiencia sólida con Python, SQL, Spark y despliegue de modelos en AWS.
"""

# Lista de modelos activos en tu cuenta (de más ligero a más potente)
modelos_disponibles = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-hydrogen"]
response = None

for modelo in modelos_disponibles:
    try:
        print(f"Probando con {modelo}...")
        response = client.models.generate_content(
            model=modelo,
            contents=f"Extrae los datos clave de esta oferta de empleo:\n{texto_oferta}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=OfertaEmpleo,
            ),
        )
        break  # Si funciona, salimos del bucle
    except Exception as error:
        print(f"  -> {modelo} saturado o no disponible, saltando al siguiente...")

if response:
    oferta = response.parsed

    print("\n--- Ficha Técnica Extraída ---")
    print("Empresa:", oferta.empresa)
    print("Puesto:", oferta.puesto)
    print("Tecnologías:", oferta.tecnologias)
    print("¿Es 100% remoto?:", oferta.remoto)
    print("Salario anual:", oferta.salario_anual)

    print("\n--- JSON crudo (response.text) ---")
    print(response.text)

    print("\n--- Consumo de Tokens ---")
    print("Total tokens:", response.usage_metadata.total_token_count)
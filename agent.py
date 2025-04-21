"""Food & Weather Buddy – Agent API (v2)
"""
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from rag import FoodRAG
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Food & Weather Buddy")
rag = FoodRAG()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o restringe a ["http://localhost:8000"]
    allow_methods=["*"],
    allow_headers=["*"],
)




class Query(BaseModel):
    query: str
    lat: float
    lon: float
    k: int = 5
    pais: str

OPEN_METEO = "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

def get_temp(lat: float, lon: float) -> float:
    r = requests.get(OPEN_METEO.format(lat=lat, lon=lon), timeout=10)
    r.raise_for_status()
    return r.json()["current_weather"]["temperature"]

@app.post("/recommend")
@app.post("/recommend/")
def recommend(body: Query):
    temp_c = get_temp(body.lat, body.lon)

    advice = (
        "Hace calor, recomendamos platos frescos 🍉🥗" if temp_c > 28 else
        "Clima templado; prueba algo reconfortante 🍲"
    )
    prompt_pais = ""
    if body.pais:
        prompt_pais = "Prioriza todas las recetas de este país: " + body.pais
    # Construir la consulta enriquecida con advice y país
    enriched_query = f"{body.query}. {advice}. { body.pais or ''}".strip()

    raw_results = rag.search(enriched_query, limit=body.k)

   
    def clean(v):
        return v if v not in (None, "None", "[]") else ""

    recs = [
        {
            "Nombre": rec.get("Nombre"),
            "URL": rec.get("URL"),
            "Ingredientes": clean(rec.get("Ingredientes")),
            "Pasos": clean(rec.get("Pasos")),
            "Pais": rec.get("Pais"),
            "Duracion": rec.get("Duracion"),
            "Categoria": clean(rec.get("Categoria")),
            "Valoracion": rec.get("Valoracion y Votos"),
            "Comensales": rec.get("Comensales"),
            "Tiempo": rec.get("Tiempo"),
            "Dificultad": rec.get("Dificultad"),
            "Valor_nutricional": rec.get("Valor nutricional"),
            "similarity": float(score),
        }
        for score, rec in raw_results
    ]

    return {"temperature_c": temp_c, "advice": advice, "recommendations": recs}

app.mount("/", StaticFiles(directory=".", html=True), name="static")
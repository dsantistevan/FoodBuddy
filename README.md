# Recetas Buddy 🍳

FastAPI + RAG que recomienda recetas del dataset **somosnlp/RecetasDeLaAbuela** y las enriquece con clima en vivo. Incluye un pequeño _frontend_ (index.html).

---
## 📦 Ejecución con Docker Compose

```bash
# 1. clona este repo y entra en la carpeta

# 2. construye y levanta
docker compose up --build
```

```
├─ agent.py            # API FastAPI + CORS + estáticos
├─ rag.py              # capa RAG (descarga dataset, genera embeddings, cache)
├─ index.html          # Frontend Tailwind + geolocalización
├─ Dockerfile          # imagen python:3.11‑slim
├─ docker-compose.yml  # expone 8000
└─ requirements.txt    # dependencias si quieres correr sin Docker
```

* El primer arranque descarga el dataset (~65 MB) y genera los embeddings; se guardan en `recetas_cached/` (≈ 150 MB).
* Navega a **http://localhost:8000/index.html** para usar la UI.

### Variables de entorno útiles
| Variable | Default | Uso |
|----------|---------|-----|
| `RECETAS_DEVICE` | *auto* | `cuda`, `mps` o `cpu` para forzar GPU/CPU |
| `RECETAS_CACHE`  | `recetas_cached` | carpeta donde se guarda el dataset procesado |
| `RECETAS_BATCH`  | `256` | tamaño de lote para embeddings |

Ejemplo:
```bash
docker compose up --build -e RECETAS_DEVICE=cuda
```

---
## ⚙️ Ejecución sin Docker

### 1. Entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```
(Se incluyen `sentence-transformers`, `datasets`, `faiss-cpu`, `fastapi`, `uvicorn`, etc.)

### 3. Primer arranque (genera embeddings y cache)
```bash
python -c "from rag import RecetasRAG; RecetasRAG()"
```

### 4. Levantar la API
```bash
uvicorn agent:app --host 0.0.0.0 --port 8000
```

Abre **http://localhost:8000/index.html** y permite la geolocalización.

---
## 📑 Endpoints
| Método | Path | Descripción |
|--------|------|-------------|
| `GET`  | `/docs` | Swagger UI |
| `POST` | `/recommend` | cuerpo JSON `{query, lat, lon, k, pais?}` y devuelve recetas + clima |

---
## 📝 Licencias
* Dataset © SomelNLP – CC BY 4.0
* Código MIT License


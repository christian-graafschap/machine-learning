# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Wat doet dit project?

**California Housing Price Predictor** — een end-to-end machine learning webapplicatie die de mediane huizenprijs in Californische wijken voorspelt.

Een gebruiker vult via een webformulier kenmerken in van een wijk (ligging, leeftijd van woningen, aantal kamers, bevolking, inkomen en nabijheid van de oceaan). De frontend stuurt die gegevens naar een REST API, die het voorspelde bedrag teruggeeft en toont.

### Onderdelen

| Onderdeel | Technologie | Rol |
|---|---|---|
| `housing-prices.py` | Python / scikit-learn | Traint het ML-model en slaat het op |
| `api/` | FastAPI + uvicorn | Laadt het model en serveert voorspellingen |
| `web/` | HTML / JS / nginx | Statische frontend; roept de API aan |
| `docker-compose.yml` | Docker | Draait api én web samen in containers |

---

## Hoe werkt het?

### 1. ML-model trainen (`housing-prices.py`)

Het script:
1. Download automatisch `housing.csv` van GitHub naar `datasets/` als het bestand nog niet bestaat.
2. Splitst de data gestratificeerd op `median_income` (80% train / 20% test).
3. Bouwt een `ColumnTransformer`-pipeline met feature-engineering:
   - **Ratio's**: bedrooms/rooms, rooms/households, population/households
   - **Log-transformatie**: scheefverdeelde kolommen (rooms, bedrooms, population, households, income)
   - **Geografische clustering**: `ClusterSimilarity` groepeert lat/lon in 10 clusters via KMeans en berekent RBF-gelijkenis
   - **Categorisch**: `ocean_proximity` via OneHotEncoder
4. Tunet een `RandomForestRegressor` met `RandomizedSearchCV` (10 iteraties, 3-fold CV).
5. Slaat het beste model op als `model/california_housing_model.pkl` via joblib.

Uitvoer in de terminal: beste hyperparameters en RMSE op de testset.

### 2. API (`api/app.py`)

FastAPI-app met twee endpoints:

- `GET /` — health check, geeft `{"message": "Housing price API is running 🚀"}`
- `POST /predict` — ontvangt een JSON-body met woningkenmerken, past het model toe en geeft de voorspelde prijs terug als `{"prediction": 452000.00}`

Het model wordt bij opstart geladen vanuit `/model/california_housing_model.pkl` (via Docker volume mount). De `safe()`-functie voorkomt `log(0)`-fouten door waarden ≤ 0 op 1 te zetten.

### 3. Frontend (`web/`)

Statische pagina (`index.html` + `app.js`) geserveerd door nginx. De formuliervelden zijn bij laden al gevuld met voorbeeldwaarden voor een wijk in de San Francisco Bay Area. Na klik op "Predict Price" verstuurt `app.js` een `POST`-request naar `http://localhost:8000/predict` en toont de voorspelling.

### 4. Gedeelde module `features.py`

`features.py` (in `api/`) definieert `column_ratio` en `ClusterSimilarity`. Beide worden gebruikt:
- door het trainingsscript (via `sys.path.insert`)
- door `api/app.py` (nodig voor het deserialiseren van het joblib-pickle)

---

## Lokaal opstarten (stap voor stap)

### Vereisten

- Python 3.11+ met pip
- Docker Desktop (draaiend)

### Stap 1 — Python-dependencies installeren

```bash
pip install -r api/requirements.txt
pip install scipy
```

> `scipy` is nodig voor `RandomizedSearchCV` in het trainingsscript maar staat niet in `requirements.txt`.

### Stap 2 — Model trainen

Het model moet **vóór** Docker bestaan. Voer uit vanuit de projectroot:

**Linux / macOS / Git Bash:**
```bash
PYTHONPATH=api python housing-prices.py
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "api"; python housing-prices.py
```

**Windows (CMD):**
```cmd
set PYTHONPATH=api && python housing-prices.py
```

Dit duurt enkele minuten (hyperparameter search). Daarna staat `model/california_housing_model.pkl` klaar.

### Stap 3 — Docker starten

```bash
docker-compose up --build
```

Of dubbelklik `run-docker.bat` op Windows.

- **Frontend:** `http://localhost:3000`
- **API:** `http://localhost:8000`
- **API-docs (Swagger):** `http://localhost:8000/docs`

### Stoppen

```bash
docker-compose down
```

---

## API lokaal testen (zonder Docker)

```bash
cd api
uvicorn app:app --reload
```

De API draait dan op `http://localhost:8000`. De frontend in `web/` werkt dan ook — open `web/index.html` direct in de browser.

Voorbeeldverzoek via curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

---

## Projectstructuur

```
housing-prices.py           ← trainingsscript (draait buiten Docker)
api/
  app.py                    ← FastAPI app, laadt model via joblib
  features.py               ← gedeelde module: column_ratio, ClusterSimilarity
  Dockerfile                ← python:3.11-slim, uvicorn op poort 8000
  requirements.txt          ← fastapi, uvicorn, pandas, scikit-learn, joblib
web/
  index.html                ← formulier
  app.js                    ← fetch-logica naar de API
  style.css                 ← opmaak
  Dockerfile                ← nginx:alpine, poort 80 → host 3000
docker-compose.yml          ← koppelt api + web, mount ./model:/model
run-docker.bat              ← Windows-shortcut voor docker-compose up --build
model/                      ← gegenereerd door trainingsscript (.gitignore)
datasets/                   ← housing.csv (.gitignore, wordt auto gedownload)
```

---

## Aandachtspunten

### Hardcoded API-URL in de frontend

`web/app.js` roept `http://localhost:8000/predict` aan. Dit werkt als de gebruiker de app in de **browser op de host** opent en poort 8000 exposed is. Voor server-side of container-interne aanroepen moet de URL veranderen naar `http://api:8000/predict`.

### Model volume mount

`docker-compose.yml` mount `./model:/model`. In `app.py` is het pad `../model/california_housing_model.pkl` relatief aan `/app`, wat uitkomt op `/model/california_housing_model.pkl`. Correct — maar het model moet lokaal bestaan vóór `docker-compose up`.

### `features.py` op twee plaatsen nodig

Omdat joblib het pickle deserialiseert met dezelfde klassen als waarmee het getraind is, moet `features.py` beschikbaar zijn in zowel het trainingsscript als de API-container. Het trainingsscript regelt dit via `sys.path.insert`; de container via `PYTHONPATH=/app` in de Dockerfile.

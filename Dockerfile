# Gebruik een lichte Python image
FROM python:3.11-slim

# Werkdirectory in de container
WORKDIR /app

# Kopieer bestanden
COPY . /app

# Installeer dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start de API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
# Image de l'API de scoring JobMatch AI — Chapitre 9.
FROM python:3.12-slim

WORKDIR /app

# Les dépendances d'abord, seules : cette couche n'est reconstruite que
# si requirements-api.txt change — pas à chaque modification du code.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Le code ensuite. Ni les tests, ni les données, ni aucun secret :
# l'image ne contient que ce que la production exécute.
COPY jobmatch/ jobmatch/

EXPOSE 8000

CMD ["uvicorn", "jobmatch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

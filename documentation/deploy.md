## Activer les API nécessaires

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

## Construire et pousser l'image sur Cloud Run

```bash
gcloud builds submit --tag gcr.io/hadock404-project/api-vision-computer
```

## Vérifier que l'image est bien disponible

```bash
gcloud container images list
```

## Déployer sur Cloud Run

```bash
gcloud run deploy api-vision-computer \
  --image gcr.io/hadock404-project/api-vision-computer \
  --platform managed \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi
```

## Adresse

Service URL: https://api-vision-computer-782672784164.northamerica-northeast1.run.app
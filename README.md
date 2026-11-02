# Almond-Sorting-
AI based almond sorting algorithm for Key Technologies

Model

## Database

Inspection results, models, and datasets are stored in PostgreSQL 18; videos,
images, and model weights live in a local media folder. For local development:

```bash
cp .env.example .env            # set a real DINOS_DB_PASSWORD
docker compose up -d
python -m app.storage migrate
```

See [docs/database.md](docs/database.md) for the schema and design decisions,
Windows installation, configuration, and the migration workflow.

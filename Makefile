.PHONY: ui pipeline mlflow-up mlflow-down mlflow-logs feast-apply feast-ui

ui:
	poetry run uvicorn ui.server:app --reload

pipeline:
	poetry run python data/pipeline.py --start $(START) --end $(END) $(if $(SYMBOL),--symbol $(SYMBOL),)

mlflow-up:
	docker compose -f docker-compose.mlflow.yml --env-file .env up -d --build

mlflow-down:
	docker compose -f docker-compose.mlflow.yml --env-file .env down

mlflow-logs:
	docker compose -f docker-compose.mlflow.yml logs -f mlflow

feast-apply:
	set -a && . .env && set +a && cd feature_store && poetry run feast --chdir feature_repo apply

feast-ui:
	set -a && . .env && set +a && cd feature_store && poetry run feast --chdir feature_repo ui

.PHONY: ui pipeline train evaluate tensorboard mlflow-up mlflow-down mlflow-logs feast-apply feast-ui

ui:
	poetry run uvicorn ui.server:app --reload

pipeline:
	poetry run python data/pipeline.py --start $(START) --end $(END) $(if $(SYMBOL),--symbol $(SYMBOL),)

train:
	set -a && . .env && set +a && poetry run python env/train.py --timesteps $(if $(TIMESTEPS),$(TIMESTEPS),50000) $(if $(N_ENVS),--n-envs $(N_ENVS),) $(if $(SEED),--seed $(SEED),) $(if $(ENT_COEF),--ent-coef $(ENT_COEF),)

evaluate:
	set -a && . .env && set +a && poetry run python env/evaluate.py $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(MLFLOW_RUN_ID),--mlflow-run-id $(MLFLOW_RUN_ID),$(if $(MODEL),--model $(MODEL),--random)) $(if $(SPLIT),--split $(SPLIT),)

tensorboard:
	poetry run tensorboard --logdir logs/ppo_tensorboard

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

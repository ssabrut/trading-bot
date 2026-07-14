.PHONY: ui pipeline train evaluate tensorboard walk-forward walk-forward-all walk-forward-summary mlflow-up mlflow-down mlflow-logs feast-apply feast-ui

ui:
	poetry run uvicorn ui.server:app --reload

pipeline:
	poetry run python data/pipeline.py --start $(START) --end $(END) $(if $(SYMBOL),--symbol $(SYMBOL),)

train:
	set -a && . .env && set +a && poetry run python env/train.py --timesteps $(if $(TIMESTEPS),$(TIMESTEPS),50000) $(if $(N_ENVS),--n-envs $(N_ENVS),) $(if $(SEED),--seed $(SEED),) $(if $(ENT_COEF),--ent-coef $(ENT_COEF),) $(if $(LR),--learning-rate $(LR),)

evaluate:
	set -a && . .env && set +a && poetry run python env/evaluate.py $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(MLFLOW_RUN_ID),--mlflow-run-id $(MLFLOW_RUN_ID),$(if $(MODEL),--model $(MODEL),--random)) $(if $(SPLIT),--split $(SPLIT),) $(if $(BEST),--best,)

tensorboard:
	poetry run tensorboard --logdir logs/ppo_tensorboard

walk-forward:
	set -a && . .env && set +a && poetry run python env/walk_forward.py --fold $(FOLD) $(if $(TIMESTEPS),--timesteps $(TIMESTEPS),) $(if $(ENT_COEF),--ent-coef $(ENT_COEF),) $(if $(LR),--learning-rate $(LR),) $(if $(SEED),--seed $(SEED),)

walk-forward-all:
	set -a && . .env && set +a && \
	N_FOLDS=$$(poetry run python -c "from env.walk_forward import FOLD_TEST_START_YEARS; print(len(FOLD_TEST_START_YEARS))") && \
	for i in $$(seq 1 $$N_FOLDS); do \
		echo "=== walk-forward fold $$i / $$N_FOLDS ==="; \
		poetry run python env/walk_forward.py --fold $$i $(if $(TIMESTEPS),--timesteps $(TIMESTEPS),) $(if $(ENT_COEF),--ent-coef $(ENT_COEF),) $(if $(LR),--learning-rate $(LR),) $(if $(SEED),--seed $(SEED),) || exit 1; \
	done && \
	poetry run python env/walk_forward.py --summary

walk-forward-summary:
	set -a && . .env && set +a && poetry run python env/walk_forward.py --summary

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

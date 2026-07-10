.PHONY: ui pipeline

ui:
	poetry run uvicorn ui.server:app --reload

pipeline:
	poetry run python data/pipeline.py --start $(START) --end $(END) $(if $(SYMBOL),--symbol $(SYMBOL),)

UV ?= uv
export UV_PROJECT_ENVIRONMENT = .venv-main
VENV = .venv-main
AAC = $(VENV)/bin/aac
RUN ?= configs/regimes/r1-chrono-full.yaml
SEED ?= 0
TABLE ?= 4.6

.PHONY: setup lock fixtures run validate report reports

setup:
	$(UV) sync --extra cpu

lock:
	$(UV) lock

fixtures:
	$(AAC) fixtures build

run:
	$(AAC) run --config $(RUN) --seed $(SEED)

validate:
	$(AAC) validate --run $(RUN_ID)

report:
	$(AAC) reports build --table $(TABLE)

reports:
	$(AAC) reports build --all

-include mk/*.mk

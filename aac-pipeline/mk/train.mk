ifeq ($(origin HF_HOME),undefined)
HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
HF_ENV =
endif
C1SMOKE_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 AAC_OUT=build/c1smoke $(HF_ENV)
C1SMOKE_TRAIN = configs/c1smoke/c1-smoke.yaml configs/c1smoke/c1-smoke-repeat.yaml
C1SMOKE_INFER = configs/c1smoke/r1-content-only-c1smoke.yaml

CPUSMOKE_REAL_ENV = AAC_OUT=build/cpusmoke-real $(HF_ENV)

.PHONY: train c1-smoke token-stats c1-cpusmoke-real

token-stats:
	$(HF_ENV) $(VENV)/bin/python -m aac.components.c1_encoder token-stats --out data/derived/averitec/token-stats.json

c1-cpusmoke-real:
	$(CPUSMOKE_REAL_ENV) $(AAC) train --config configs/c1smoke/c1-chrono-cpusmoke.yaml --seed 0 --force
	$(CPUSMOKE_REAL_ENV) $(AAC) run --config configs/c1smoke/r1-chrono-content-only-cpusmoke.yaml --seed 0 --force

train:
	$(HF_ENV) $(AAC) train --config $(RUN) --seed $(SEED)

c1-smoke:
	@for cfg in $(C1SMOKE_TRAIN); do \
 echo "== c1-smoke train $$cfg seed 0"; \
 $(C1SMOKE_ENV) $(AAC) train --config $$cfg --seed 0 --force || exit 1; \
	done
	@for cfg in $(C1SMOKE_TRAIN); do \
 rid=$$($(C1SMOKE_ENV) $(AAC) run-id --config $$cfg --seed 0) || exit 1; \
 echo "c1-smoke checkpoint $$rid $$(jq -r .checkpoint_sha256 build/c1smoke/runs/$$rid/model.json)"; \
	done
	$(C1SMOKE_ENV) $(AAC) run --config $(C1SMOKE_INFER) --seed 0 --force
	$(C1SMOKE_ENV) $(AAC) validate --run $$($(C1SMOKE_ENV) $(AAC) run-id --config $(C1SMOKE_INFER) --seed 0)

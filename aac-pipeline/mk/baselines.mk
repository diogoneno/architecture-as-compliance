ifeq ($(origin HF_HOME),undefined)
BL_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
BL_HF_ENV =
endif
BL_PY = $(VENV)/bin/python
BASELINE_FLAGS ?= --reuse-indexed
BL_RUN = $(BL_HF_ENV) $(BL_PY) -m aac.baselines.external run --seed $(SEED) $(BASELINE_FLAGS)
BL_ATTACK = bodega-fc-bertattack bodega-rd-bertattack
BL_ATTACK_UNTARGETED = bodega-fc-bertattack-untargeted bodega-rd-bertattack-untargeted
BL_RD_CHUNKS = $(sort $(wildcard configs/baselines/bodega-rd-bertattack-untargeted-c*.yaml))

.PHONY: baseline baselines-clean baselines-attack baselines-attack-untargeted baselines-attack-rd-chunks baseline-averitec

baseline:
	$(BL_RUN) --config $(RUN)

baselines-clean:
	$(BL_RUN) --config configs/baselines/bodega-fc-clean.yaml --config configs/baselines/bodega-rd-clean.yaml

baselines-attack:
	@d=$$(mktemp -d /tmp/aac-bl.XXXXXX); pids=""; for n in $(BL_ATTACK); do \
 nohup env $(BL_RUN) --config configs/baselines/$$n.yaml > "$$d/$$n.log" 2>&1 & pids="$$pids $$!"; \
 echo "baselines-attack: $$n started (pid $$!)"; \
	done; \
	status=0; for p in $$pids; do wait $$p || status=1; done; \
	for n in $(BL_ATTACK); do echo "--- $$n"; cat "$$d/$$n.log"; done; \
	if [ "$$status" -eq 0 ]; then rm -rf "$$d"; else echo "baselines-attack: failed; logs kept in $$d" >&2; fi; \
	exit $$status

baselines-attack-untargeted:
	$(MAKE) -s baselines-attack BL_ATTACK="$(BL_ATTACK_UNTARGETED)"

baselines-attack-rd-chunks:
	@test -n "$(BL_RD_CHUNKS)" || { echo "baselines-attack-rd-chunks: no chunk configs (python scripts/make_rd_chunk_configs.py)" >&2; exit 2; }
	@$(BL_PY) scripts/make_rd_chunk_configs.py --check
	@d=$$(mktemp -d /tmp/aac-bl.XXXXXX); \
	echo "baselines-attack-rd-chunks: $(words $(BL_RD_CHUNKS)) chunks, sequential; log $$d/rd-chunks.log"; \
	env PYTHONUNBUFFERED=1 $(BL_RUN) --fail-fast $(addprefix --config ,$(BL_RD_CHUNKS)) > "$$d/rd-chunks.log" 2>&1; \
	status=$$?; cat "$$d/rd-chunks.log"; \
	if [ "$$status" -eq 0 ]; then rm -rf "$$d"; else echo "baselines-attack-rd-chunks: failed; log kept in $$d" >&2; fi; \
	exit $$status

baseline-averitec:
	$(BL_RUN) --config configs/baselines/averitec-gold-baseline.yaml

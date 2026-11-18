MATRIX ?= configs/run-matrix.yaml

.PHONY: mini-runs

mini-runs:
	$(SMOKE_ENV) $(AAC) fixtures build
	$(SMOKE_ENV) $(AAC) matrix prune --matrix $(MATRIX)
	$(SMOKE_ENV) $(AAC) matrix list --matrix $(MATRIX) | while read cfg seed rid; do \
 echo "== $$cfg seed $$seed ($$rid)"; \
 $(SMOKE_ENV) $(AAC) run --config $$cfg --seed $$seed --force || exit 1; \
 $(SMOKE_ENV) $(AAC) validate --run $$rid >/dev/null || exit 1; \
	done
	$(SMOKE_ENV) $(AAC) index rebuild --check-matrix $(MATRIX)

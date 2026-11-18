ifeq ($(origin HF_HOME),undefined)
BENCH_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
BENCH_HF_ENV =
endif
BENCH_PY = $(VENV)/bin/python
BENCH_SMOKE_OUT = build/bench-smoke
BENCH_SMOKE_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 AAC_OUT=$(BENCH_SMOKE_OUT) $(BENCH_HF_ENV)
BENCH_SMOKE_C1 = configs/c1smoke/c1-smoke.yaml
BENCH_REPORT_ARGS ?=

.PHONY: bench bench-smoke bench-report

bench:
	$(BENCH_HF_ENV) $(BENCH_PY) -m aac.bench.retrieval run --config $(RUN) --seed $(SEED)

bench-smoke:
	$(BENCH_SMOKE_ENV) $(BENCH_PY) -m aac.bench.retrieval make-smoke-store --out $(BENCH_SMOKE_OUT)/store
	@rid=$$($(BENCH_SMOKE_ENV) $(AAC) run-id --config $(BENCH_SMOKE_C1) --seed 0) || exit 1; \
	if test -f $(BENCH_SMOKE_OUT)/runs/$$rid/SEALED; then echo "bench-smoke: reusing sealed $$rid (c1-smoke seed 0)"; \
	else $(BENCH_SMOKE_ENV) $(AAC) train --config $(BENCH_SMOKE_C1) --seed 0 --force || exit 1; fi
	$(BENCH_SMOKE_ENV) $(BENCH_PY) -m aac.bench.retrieval run --config configs/bench/smoke.yaml --seed 0 --force
	$(BENCH_SMOKE_ENV) $(BENCH_PY) -m aac.bench.report --bench bench-smoke --seed 0 \
 --c1-pair c1-smoke c1-smoke-repeat --r1-pair r1-content-only-c1smoke r1-content-only-c1smoke-repeat

bench-report:
	$(BENCH_PY) -m aac.bench.report $(BENCH_REPORT_ARGS)

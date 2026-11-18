ifeq ($(origin HF_HOME),undefined)
EV_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
EV_HF_ENV =
endif
EV_PY = $(VENV)/bin/python
EV_SMOKE_SENTINELS = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0
EV_SMOKE_SL = configs/tinyworld/sl-tw.yaml
EV_SMOKE_CONFIG = configs/tinyworld/ev-tw.yaml
EV_SL_THREADS_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EV_CPUSMOKE_CONFIG = configs/retrieval/evidence-cpusmoke.yaml

.PHONY: evidence evidence-cpusmoke evidence-smoke

evidence:
	$(EV_HF_ENV) $(EV_PY) -m aac.components.c2_evidence_run run --config $(RUN) --seed $(SEED)

evidence-cpusmoke:
	CUDA_VISIBLE_DEVICES= $(EV_HF_ENV) $(EV_PY) -m aac.components.c2_evidence_run run --config $(EV_CPUSMOKE_CONFIG) \
 --seed 0 --force

evidence-smoke: tinyworld
	@set -e; first=; for out in build/tw-ev build/tw-ev2; do \
 rid=$$($(EV_SMOKE_SENTINELS) AAC_OUT=$$out $(AAC) run-id --config $(EV_SMOKE_SL) --seed 0); \
 if test -f $$out/runs/$$rid/SEALED; then echo "evidence-smoke: reusing sealed $$rid in $$out"; \
 else $(EV_SMOKE_SENTINELS) AAC_OUT=$$out $(EV_SL_THREADS_ENV) $(EV_PY) -m aac.data.shortlist run \
 --config $(EV_SMOKE_SL) --seed 0 --workers 2 >&2; fi; \
 line=$$($(EV_SMOKE_SENTINELS) AAC_OUT=$$out CUDA_VISIBLE_DEVICES= $(EV_HF_ENV) $(EV_PY) -m aac.components.c2_evidence_run \
 run --config $(EV_SMOKE_CONFIG) --seed 0 --force | tee /dev/stderr \
 | sed -n 's/^evidence: run_id=\(EV-[0-9a-f]*-s[0-9]*\) evidence_sha256=\([0-9a-f]*\)$$/\1 \2/p'); \
 test -n "$$line" || { echo "evidence-smoke: $$out printed no run_id/evidence_sha256 line" >&2; exit 1; }; \
 echo "evidence-smoke: $$line"; \
 if test -z "$$first"; then first=$$line; \
 else test "$$first" = "$$line" || { echo "evidence-smoke: FAIL: build/tw-ev gave '$$first', build/tw-ev2 gave '$$line'" >&2; exit 1; }; fi; \
	done; \
	echo "evidence-smoke: build/tw-ev and build/tw-ev2 agree"

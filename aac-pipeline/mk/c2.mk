ifeq ($(origin HF_HOME),undefined)
C2_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
C2_HF_ENV =
endif
C2_PY = $(VENV)/bin/python
C2_OUT = build/tw-c2
C2_SENTINELS = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0
C2_SL = configs/tinyworld/sl-tw.yaml
C2_EV = configs/tinyworld/ev-tw.yaml
C2_EV_RD = configs/tinyworld/ev-rd-tw.yaml
C2_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.PHONY: c2-smoke

c2-smoke: tinyworld
	@set -e; out=$(C2_OUT); \
	rid=$$($(C2_SENTINELS) AAC_OUT=$$out $(AAC) run-id --config $(C2_SL) --seed 0); \
	if test -f $$out/runs/$$rid/SEALED; then echo "c2-smoke: reusing sealed $$rid"; \
	else $(C2_SENTINELS) AAC_OUT=$$out $(C2_CPU_ENV) $(C2_PY) -m aac.data.shortlist run --config $(C2_SL) --seed 0 \
 --workers 2 >&2; fi; \
	rid=$$($(C2_SENTINELS) AAC_OUT=$$out $(AAC) run-id --config $(C2_EV) --seed 0); \
	if test -f $$out/runs/$$rid/SEALED; then echo "c2-smoke: reusing sealed $$rid"; \
	else $(C2_SENTINELS) AAC_OUT=$$out CUDA_VISIBLE_DEVICES= $(C2_HF_ENV) $(C2_PY) -m aac.components.c2_evidence_run run \
 --config $(C2_EV) --seed 0 >&2; fi; \
	rid=$$($(C2_SENTINELS) AAC_OUT=$$out $(AAC) run-id --config $(C2_EV_RD) --seed 0); \
	if test -f $$out/runs/$$rid/SEALED; then echo "c2-smoke: reusing sealed $$rid"; \
	else $(C2_SENTINELS) AAC_OUT=$$out CUDA_VISIBLE_DEVICES= $(C2_HF_ENV) $(C2_PY) -m aac.components.c2_evidence_run run \
 --config $(C2_EV_RD) --seed 0 >&2; fi; \
	$(C2_SENTINELS) AAC_OUT=$$out CUDA_VISIBLE_DEVICES= $(C2_HF_ENV) $(C2_PY) -m aac.components.c2_retrieval smoke \
 --ev-name ev-tw --rd-ev-name ev-rd-tw --fc-items tests/fixtures/corpus/fc-fx.json

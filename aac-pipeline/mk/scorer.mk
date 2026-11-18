ifeq ($(origin HF_HOME),undefined)
SC_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
SC_HF_ENV =
endif
SC_PY = $(VENV)/bin/python
SC_OUT = build/tw-c2s
SC_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 AAC_OUT=$(SC_OUT)
SC_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SC_SL = configs/tinyworld/sl-tw.yaml
SC_EV = configs/tinyworld/ev-tw.yaml
SC_CONFIG = configs/tinyworld/c2s-tw.yaml

.PHONY: scorer-smoke

scorer-smoke: tinyworld
	@set -e; \
	rid=$$($(SC_ENV) $(AAC) run-id --config $(SC_SL) --seed 0); \
	if test -f $(SC_OUT)/runs/$$rid/SEALED; then echo "scorer-smoke: reusing sealed $$rid" >&2; \
	else $(SC_ENV) $(SC_CPU_ENV) $(SC_PY) -m aac.data.shortlist run --config $(SC_SL) --seed 0 --workers 2 >&2; fi; \
	rid=$$($(SC_ENV) $(AAC) run-id --config $(SC_EV) --seed 0); \
	if test -f $(SC_OUT)/runs/$$rid/SEALED; then echo "scorer-smoke: reusing sealed $$rid" >&2; \
	else $(SC_ENV) CUDA_VISIBLE_DEVICES= $(SC_HF_ENV) $(SC_PY) -m aac.components.c2_evidence_run run --config $(SC_EV) \
 --seed 0 >&2; fi; \
	$(SC_ENV) CUDA_VISIBLE_DEVICES= $(SC_HF_ENV) $(AAC) train --config $(SC_CONFIG) --seed 0 --force >&2; \
	crid=$$($(SC_ENV) $(AAC) run-id --config $(SC_CONFIG) --seed 0); \
	$(SC_ENV) CUDA_VISIBLE_DEVICES= $(SC_HF_ENV) $(SC_PY) -m aac.components.c2_scorer smoke --run-id $$crid

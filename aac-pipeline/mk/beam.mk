.PHONY: beam-smoke

BEAM_OUT = build/tw-beam
BEAM_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=$(BEAM_OUT)
BEAM_CONFIG = configs/tinyworld/r3-tw-beam.yaml

beam-smoke: tinyworld
	@set -e; \
	rid=$$($(BEAM_ENV) $(AAC) run-id --config $(FC_SL) --seed 0); \
	if test -f $(BEAM_OUT)/runs/$$rid/SEALED; then echo "beam-smoke: reusing sealed $$rid" >&2; \
	else $(BEAM_ENV) $(FC_CPU_ENV) $(FC_PY) -m aac.data.shortlist run --config $(FC_SL) --seed 0 --workers 2 >&2; fi; \
	rid=$$($(BEAM_ENV) $(AAC) run-id --config $(FC_EV) --seed 0); \
	if test -f $(BEAM_OUT)/runs/$$rid/SEALED; then echo "beam-smoke: reusing sealed $$rid" >&2; \
	else $(BEAM_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(FC_PY) -m aac.components.c2_evidence_run run --config $(FC_EV) \
 --seed 0 >&2; fi; \
	for cfg in $(FC_C1) $(FC_C2S); do \
 rid=$$($(BEAM_ENV) $(AAC) run-id --config $$cfg --seed 0); \
 if test -f $(BEAM_OUT)/runs/$$rid/SEALED; then echo "beam-smoke: reusing sealed $$rid" >&2; \
 else $(BEAM_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) train --config $$cfg --seed 0 >&2; fi; \
	done; \
	rid=$$($(BEAM_ENV) $(AAC) run-id --config $(FC_R1) --seed 0); \
	if test -f $(BEAM_OUT)/runs/$$rid/SEALED; then echo "beam-smoke: reusing sealed $$rid" >&2; \
	else $(BEAM_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) run --config $(FC_R1) --seed 0 >&2; fi; \
	$(BEAM_ENV) CUDA_VISIBLE_DEVICES= $(MAKE) --no-print-directory attack RUN=$(BEAM_CONFIG) SEED=0 ATTACK_FORCE=1 >&2; \
	rid=$$($(BEAM_ENV) $(AAC) run-id --config $(BEAM_CONFIG) --seed 0); \
	$(BEAM_ENV) $(AAC) validate --run $$rid >/dev/null; \
	$(BEAM_ENV) $(AAC) ledger-check $(BEAM_OUT)/runs/$$rid >/dev/null; \
	echo "beam-smoke: $$rid L2 ok"

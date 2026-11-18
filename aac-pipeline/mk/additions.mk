ifeq ($(origin HF_HOME),undefined)
GA_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
GA_HF_ENV = HF_HUB_OFFLINE=1
endif
GA_PY = $(VENV)/bin/python
GA_OUT = build/tw-gate
GA_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=$(GA_OUT)
GA_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GA_TW_SL = configs/tinyworld/sl-tw.yaml
GA_TW_EV = configs/tinyworld/ev-tw.yaml
GA_TW_C1 = configs/tinyworld/c1-tw.yaml
GA_TW_C2S = configs/tinyworld/c2s-tw.yaml
GA_TW_R1 = configs/tinyworld/r1-tw-full.yaml
GA_TW_R3 = configs/tinyworld/r3-tw-gate-aware.yaml

define GA_LEDGER_PY
import collections, json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = [json.loads(x) for x in (d / "attack-ledger.jsonl").read_text(encoding="ascii").splitlines if x]
rec = json.loads((d / "run-record.json").read_text(encoding="utf-8"))
ends = collections.Counter(r["end_state"] for r in rows)
gates = collections.Counter(str(r["posthoc_gate"]) for r in rows)
print("gate-aware-smoke-ledger: " + d.name + " rows=" + str(len(rows)) + " attacked=" + str(sum(r["queries_used"] > 0 for r in rows))
 + " end_states=" + ",".join(k + ":" + str(v) for k, v in sorted(ends.items))
 + " posthoc=" + ",".join(k + ":" + str(v) for k, v in sorted(gates.items))
 + " records=" + str(rec["record_count"]) + " depends_on=" + ",".join(rec["depends_on"]))
endef
gate-aware-smoke: export GA_LEDGER = $(GA_LEDGER_PY)

.PHONY: gate-aware-smoke

gate-aware-smoke: tinyworld
	@set -e; \
	reuse { rid=$$($(GA_ENV) $(AAC) run-id --config $$1 --seed 0); test -f $(GA_OUT)/runs/$$rid/SEALED && \
 { echo "gate-aware-smoke: reusing sealed $$rid" >&2; return 0; }; return 1; }; \
	reuse $(GA_TW_SL) || $(GA_ENV) $(GA_CPU_ENV) $(GA_PY) -m aac.data.shortlist run --config $(GA_TW_SL) --seed 0 --workers 2 >&2; \
	reuse $(GA_TW_EV) || $(GA_ENV) CUDA_VISIBLE_DEVICES= $(GA_HF_ENV) $(GA_PY) -m aac.components.c2_evidence_run run \
 --config $(GA_TW_EV) --seed 0 >&2; \
	for cfg in $(GA_TW_C1) $(GA_TW_C2S); do \
 reuse $$cfg || $(GA_ENV) CUDA_VISIBLE_DEVICES= $(GA_HF_ENV) $(AAC) train --config $$cfg --seed 0 >&2; \
	done; \
	reuse $(GA_TW_R1) || $(GA_ENV) CUDA_VISIBLE_DEVICES= $(GA_HF_ENV) $(AAC) run --config $(GA_TW_R1) --seed 0 >&2; \
	$(GA_ENV) CUDA_VISIBLE_DEVICES= $(MAKE) --no-print-directory attack RUN=$(GA_TW_R3) SEED=0 ATTACK_FORCE=1 >&2; \
	rid=$$($(GA_ENV) $(AAC) run-id --config $(GA_TW_R3) --seed 0); \
	$(GA_ENV) $(AAC) validate --run $$rid >&2; \
	$(GA_ENV) $(AAC) ledger-check $(GA_OUT)/runs/$$rid >&2; \
	$(GA_PY) -c "$$GA_LEDGER" $(GA_OUT)/runs/$$rid >&2; \
	echo "gate-aware-smoke: $$rid L2 ok"

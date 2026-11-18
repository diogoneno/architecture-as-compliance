ifeq ($(origin HF_HOME),undefined)
R4_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
R4_HF_ENV = HF_HUB_OFFLINE=1
endif
R4_PY = $(VENV)/bin/python
R4_OUT = build/tw-r4
R4_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=$(R4_OUT)
R4_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R4_TW_SL = configs/tinyworld/sl-tw.yaml
R4_TW_EV = configs/tinyworld/ev-tw.yaml
R4_TW_C1 = configs/tinyworld/c1-tw.yaml
R4_TW_C2S = configs/tinyworld/c2s-tw.yaml
R4_TW_R1 = configs/tinyworld/r1-tw-full.yaml
R4_TW_R3 = configs/tinyworld/r3-tw-pilot.yaml
R4_TW_METRIC = configs/tinyworld/r4-tw-no-calibration.yaml
R4_TW_REPLAY = configs/tinyworld/r4-tw-no-calibration-replay.yaml

define R4_LEDGER_PY
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = [json.loads(x) for x in (d / "attack-ledger.jsonl").read_text(encoding="ascii").splitlines if x]
rec = json.loads((d / "run-record.json").read_text(encoding="utf-8"))
den = [r for r in rows if r["queries_used"] == 1]
finals = sum(len(r["record_ids"]) == 2 for r in rows)
print("r4-smoke-replay: " + d.name + " rows=" + str(len(rows)) + " denominator=" + str(len(den)) + " finals=" + str(finals)
 + " successes=" + str(sum(r["end_state"] == "success" for r in den)) + " deferred=" + str(sum(r["end_state"] == "deferred" for r in den))
 + " records=" + str(rec["record_count"]) + " depends_on=" + ",".join(rec["depends_on"]))
endef
r4-smoke: export R4_LEDGER = $(R4_LEDGER_PY)

define R4_TABLE_PY
import csv, sys
lines = [x for x in open(sys.argv[1], encoding="utf-8").read.splitlines if not x.startswith("#")]
print("r4-table: " + str(len(list(csv.DictReader(lines)))) + " rows")
endef
r4-smoke: export R4_TABLE = $(R4_TABLE_PY)

.PHONY: r4-smoke

r4-smoke: tinyworld
	@set -e; \
	reuse { rid=$$($(R4_ENV) $(AAC) run-id --config $$1 --seed 0); test -f $(R4_OUT)/runs/$$rid/SEALED && \
 { echo "r4-smoke: reusing sealed $$rid" >&2; return 0; }; return 1; }; \
	reuse $(R4_TW_SL) || $(R4_ENV) $(R4_CPU_ENV) $(R4_PY) -m aac.data.shortlist run --config $(R4_TW_SL) --seed 0 --workers 2 >&2; \
	reuse $(R4_TW_EV) || $(R4_ENV) CUDA_VISIBLE_DEVICES= $(R4_HF_ENV) $(R4_PY) -m aac.components.c2_evidence_run run \
 --config $(R4_TW_EV) --seed 0 >&2; \
	for cfg in $(R4_TW_C1) $(R4_TW_C2S); do \
 reuse $$cfg || $(R4_ENV) CUDA_VISIBLE_DEVICES= $(R4_HF_ENV) $(AAC) train --config $$cfg --seed 0 >&2; \
	done; \
	reuse $(R4_TW_R1) || $(R4_ENV) CUDA_VISIBLE_DEVICES= $(R4_HF_ENV) $(AAC) run --config $(R4_TW_R1) --seed 0 >&2; \
	$(R4_ENV) CUDA_VISIBLE_DEVICES= $(MAKE) --no-print-directory attack RUN=$(R4_TW_R3) SEED=0 ATTACK_FORCE=1 >&2; \
	r3=$$($(R4_ENV) $(AAC) run-id --config $(R4_TW_R3) --seed 0); \
	$(R4_ENV) $(AAC) ledger-check $(R4_OUT)/runs/$$r3 >&2; \
	for cfg in $(R4_TW_METRIC) $(R4_TW_REPLAY); do \
 $(R4_ENV) CUDA_VISIBLE_DEVICES= $(R4_HF_ENV) $(AAC) run --config $$cfg --seed 0 --force >&2; \
 rid=$$($(R4_ENV) $(AAC) run-id --config $$cfg --seed 0); \
 $(R4_ENV) $(AAC) validate --run $$rid >&2; \
 case $$cfg in *-replay.yaml) replay=$$rid ;; *) metric=$$rid ;; esac; \
	done; \
	$(R4_ENV) $(AAC) ledger-check $(R4_OUT)/runs/$$replay >&2; \
	$(R4_PY) -c "$$R4_LEDGER" $(R4_OUT)/runs/$$replay >&2; \
	echo "r4-smoke: metric $$metric replay $$replay L2 ok"; \
	$(R4_ENV) $(AAC) reports build --table 4.5 --allow-pending >&2; \
	$(R4_PY) -c "$$R4_TABLE" $(R4_OUT)/reports/table-4.5-ablations.csv

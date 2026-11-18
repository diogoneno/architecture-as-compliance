ifeq ($(origin HF_HOME),undefined)
FC_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1
else
FC_HF_ENV =
endif
FC_PY = $(VENV)/bin/python
FC_OUT = build/tw-full
FC_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 AAC_OUT=$(FC_OUT)
FC_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FC_SL = configs/tinyworld/sl-tw.yaml
FC_EV = configs/tinyworld/ev-tw.yaml
FC_C1 = configs/tinyworld/c1-tw.yaml
FC_C2S = configs/tinyworld/c2s-tw.yaml
FC_R1 = configs/tinyworld/r1-tw-full.yaml
FC_PAUSE_CFG = configs/tinyworld/r1-tw-full-pause.yaml

define FC_LINE_PY
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rec = json.loads((d / "run-record.json").read_text(encoding="utf-8"))
sv = dict(x.split("=", 1) for x in (d / "schema-validation.txt").read_text(encoding="ascii").split if "=" in x)
names = [("fusion", "fusion.json"), ("calibration", "calibration.json"), ("gate", "gate.json"), ("floor", "evidence-floor.json")]
fitted = ",".join(n for n, f in names if (d / f).is_file)
bad = [k for k in ("violations", "records_with_claim_text", "records_with_user_id") if sv.get(k) != "0"]
if bad or fitted != "fusion,calibration,gate,floor":
 sys.exit(sys.argv[2] + ": FAIL " + d.name + " counts not 0: " + str(bad) + " fitted=" + fitted)
print(sys.argv[2] + ": " + d.name + " records=" + str(rec["record_count"]) + " violations=0 fitted=" + fitted)
endef
fullchain-smoke: export FC_LINE = $(FC_LINE_PY)

define FC_PAUSE_PY
import gzip, json, sys, pathlib
d = pathlib.Path(sys.argv[1])
recs = [json.loads(x) for f in sorted((d / "decision-log").glob("part-*.jsonl.gz")) for x in gzip.open(f, "rt", encoding="utf-8")]
test = [r for r in recs if r["split"] == "test"]
share = 100.0 * sum(bool(r["abstention_flag"]) for r in test) / len(test) if test else 0.0
line = sys.argv[2] + ": " + d.name + " deferred=" + format(share, ".1f") + "%"
print(line)
if share != 100.0:
 sys.exit(sys.argv[2] + ": FAIL not every test item deferred")
endef
fullchain-smoke: export FC_PAUSE = $(FC_PAUSE_PY)

.PHONY: fullchain-smoke

fullchain-smoke: tinyworld
	@set -e; \
	rid=$$($(FC_ENV) $(AAC) run-id --config $(FC_SL) --seed 0); \
	if test -f $(FC_OUT)/runs/$$rid/SEALED; then echo "fullchain-smoke: reusing sealed $$rid" >&2; \
	else $(FC_ENV) $(FC_CPU_ENV) $(FC_PY) -m aac.data.shortlist run --config $(FC_SL) --seed 0 --workers 2 >&2; fi; \
	rid=$$($(FC_ENV) $(AAC) run-id --config $(FC_EV) --seed 0); \
	if test -f $(FC_OUT)/runs/$$rid/SEALED; then echo "fullchain-smoke: reusing sealed $$rid" >&2; \
	else $(FC_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(FC_PY) -m aac.components.c2_evidence_run run --config $(FC_EV) \
 --seed 0 >&2; fi; \
	for cfg in $(FC_C1) $(FC_C2S); do \
 rid=$$($(FC_ENV) $(AAC) run-id --config $$cfg --seed 0); \
 if test -f $(FC_OUT)/runs/$$rid/SEALED; then echo "fullchain-smoke: reusing sealed $$rid" >&2; \
 else $(FC_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) train --config $$cfg --seed 0 >&2; fi; \
	done; \
	$(FC_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) run --config $(FC_R1) --seed 0 --force >&2; \
	rid=$$($(FC_ENV) $(AAC) run-id --config $(FC_R1) --seed 0); \
	$(FC_ENV) $(AAC) validate --run $$rid >&2; \
	$(FC_PY) -c "$$FC_LINE" $(FC_OUT)/runs/$$rid fullchain-smoke; \
	$(FC_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) run --config $(FC_PAUSE_CFG) --seed 0 --force >&2; \
	prid=$$($(FC_ENV) $(AAC) run-id --config $(FC_PAUSE_CFG) --seed 0); \
	$(FC_ENV) $(AAC) validate --run $$prid >&2; \
	$(FC_PY) -c "$$FC_LINE" $(FC_OUT)/runs/$$prid fullchain-pause-fits >&2; \
	$(FC_PY) -c "$$FC_PAUSE" $(FC_OUT)/runs/$$prid fullchain-pause

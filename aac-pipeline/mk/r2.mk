ifeq ($(origin HF_HOME),undefined)
R2_HF_HOME = $(CURDIR)/.hf
else
R2_HF_HOME = $(HF_HOME)
endif
R2_PY = $(VENV)/bin/python
R2_OUT = build/tw-r2
R2_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 AAC_OUT=$(R2_OUT)
R2_TW_HF = HF_HOME=$(R2_HF_HOME) HF_HUB_OFFLINE=1
R2_CPU_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R2_TW_SL = configs/tinyworld/sl-tw.yaml
R2_TW_EV = configs/tinyworld/ev-tw.yaml
R2_TW_C1 = configs/tinyworld/c1-tw.yaml
R2_TW_C2S = configs/tinyworld/c2s-tw.yaml
R2_TW_R1 = configs/tinyworld/r1-tw-full.yaml
R2_TW_EVRD = configs/tinyworld/ev-rd-tw.yaml
R2_TW_FC = configs/tinyworld/r2-tw-fc-full.yaml
R2_TW_RD = configs/tinyworld/r2-tw-rd-full.yaml
R2_CONFIGS ?= configs/regimes/real/r2-fc-content-only.yaml configs/regimes/real/r2-fc-full.yaml \
 configs/regimes/real/r2-rd-content-only.yaml configs/regimes/real/r2-rd-full.yaml
R2_SEEDS ?= 0 1 2 3 4
R2_EVRD_CONFIG = configs/retrieval/ev-rd-pooled.yaml
R2_SHA ?= $(shell git rev-parse HEAD)
R2_REL = $(CURDIR)/build/r2-release/$(R2_SHA)
R2_REL_ENV = AAC_ROOT=$(R2_REL) AAC_GIT_SHA=$(R2_SHA) PYTHONPATH=$(R2_REL)/src HF_HOME=$(R2_HF_HOME) HF_HUB_OFFLINE=1 \
 CUDA_VISIBLE_DEVICES=
R2_AAC = $(CURDIR)/$(VENV)/bin/aac
R2_RPY = $(CURDIR)/$(VENV)/bin/python

define R2_LINE_PY
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rec = json.loads((d / "run-record.json").read_text(encoding="utf-8"))
sv = dict(x.split("=", 1) for x in (d / "schema-validation.txt").read_text(encoding="ascii").split if "=" in x)
bad = [k for k in ("violations", "records_with_claim_text", "records_with_user_id") if sv.get(k) != "0"]
names = [("fusion", "fusion.json"), ("calibration", "calibration.json"), ("gate", "gate.json"), ("floor", "evidence-floor.json")]
fitted = ",".join(n for n, f in names if (d / f).is_file)
if bad or (len(sys.argv) > 3 and fitted != "fusion,calibration,gate,floor"):
 sys.exit(sys.argv[2] + ": FAIL " + d.name + " counts not 0: " + str(bad) + " fitted=" + fitted)
trunc = {k: v for k, v in rec["items"].items if k.startswith("scorer_truncated")}
print(sys.argv[2] + ": " + d.name + " records=" + str(rec["record_count"]) + " violations=0 claim_text=0 user_id=0"
 + (" fitted=" + fitted if fitted else "") + "".join(" " + k + "=" + str(v) for k, v in sorted(trunc.items)))
endef
define R2_FIND_PY
import sys, pathlib
from aac.run.index import StaleRun, current_sealed_run
try:
 rid = current_sealed_run(pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]))
except StaleRun as exc:
 sys.exit("refusing to reuse: " + str(exc))
if rid:
 print(rid)
endef
r2-smoke r2-runs ev-rd: export R2_LINE = $(R2_LINE_PY)
r2-smoke r2-runs ev-rd: export R2_FIND = $(R2_FIND_PY)

.PHONY: r2-smoke pooled-bundle-pull r2-release ev-rd r2-runs

r2-smoke: tinyworld
	@set -e; \
	reuse { rid=$$($(R2_ENV) $(AAC) run-id --config $$1 --seed 0); test -f $(R2_OUT)/runs/$$rid/SEALED && \
 { echo "r2-smoke: reusing sealed $$rid" >&2; return 0; }; return 1; }; \
	reuse $(R2_TW_SL) || $(R2_ENV) $(R2_CPU_ENV) $(R2_PY) -m aac.data.shortlist run --config $(R2_TW_SL) --seed 0 --workers 2 >&2; \
	reuse $(R2_TW_EV) || $(R2_ENV) CUDA_VISIBLE_DEVICES= $(R2_TW_HF) $(R2_PY) -m aac.components.c2_evidence_run run \
 --config $(R2_TW_EV) --seed 0 >&2; \
	for cfg in $(R2_TW_C1) $(R2_TW_C2S); do \
 reuse $$cfg || $(R2_ENV) CUDA_VISIBLE_DEVICES= $(R2_TW_HF) $(AAC) train --config $$cfg --seed 0 >&2; \
	done; \
	reuse $(R2_TW_R1) || $(R2_ENV) CUDA_VISIBLE_DEVICES= $(R2_TW_HF) $(AAC) run --config $(R2_TW_R1) --seed 0 >&2; \
	reuse $(R2_TW_EVRD) || $(R2_ENV) CUDA_VISIBLE_DEVICES= $(R2_TW_HF) $(R2_PY) -m aac.components.c2_evidence_run run \
 --config $(R2_TW_EVRD) --seed 0 >&2; \
	for cfg in $(R2_TW_FC) $(R2_TW_RD); do \
 $(R2_ENV) CUDA_VISIBLE_DEVICES= $(R2_TW_HF) $(AAC) run --config $$cfg --seed 0 --force >&2; \
 rid=$$($(R2_ENV) $(AAC) run-id --config $$cfg --seed 0); \
 $(R2_ENV) $(AAC) validate --run $$rid >&2; \
 $(R2_PY) -c "$$R2_LINE" $(R2_OUT)/runs/$$rid r2-smoke-run fitted >&2; \
 case $$cfg in *fc-full*) fc=$$rid ;; *) rd=$$rid ;; esac; \
	done; \
	echo "r2-smoke: fc $$fc rd $$rd violations=0"

pooled-bundle-pull:
	@test -n "$(EV)" || { echo "pooled-bundle-pull: set EV=<EV run id>" >&2; exit 1; }
	bash scripts/pull_private_bundle.sh $(EV)

r2-release:
	@set -e; sha=$(R2_SHA); rel=$(R2_REL); \
	echo "$$sha" | grep -Eq '^[0-9a-f]{40}$$' || { echo "r2-release: R2_SHA '$$sha' is not a 40-hex commit" >&2; exit 1; }; \
	if test "$$(cat $$rel/AAC_GIT_SHA 2>/dev/null)" = "$$sha"; then echo "r2-release: reusing $$rel" >&2; exit 0; fi; \
	git diff --quiet "$$sha" -- data/derived || { echo "r2-release: data/derived differs from $$sha" >&2; exit 1; }; \
	test -z "$$(git status --porcelain -- data/derived)" || { echo "r2-release: data/derived has changes" >&2; exit 1; }; \
	stage=$$rel.staging; rm -rf "$$stage"; mkdir -p "$$stage"; \
	git -C .. archive --format=tar "$$sha" aac-pipeline | tar -x --strip-components=1 -C "$$stage"; \
	rm -rf "$$stage/runs" "$$stage/data/derived" "$$stage/data/raw/averitec" "$$stage/data/raw/incrediblae" "$$stage/.hf"; \
	ln -s $(CURDIR)/runs "$$stage/runs"; ln -s $(CURDIR)/data/derived "$$stage/data/derived"; \
	ln -s $(CURDIR)/data/raw/averitec "$$stage/data/raw/averitec"; ln -s $(CURDIR)/data/raw/incrediblae "$$stage/data/raw/incrediblae"; \
	ln -s $(R2_HF_HOME) "$$stage/.hf"; \
	printf '%s\n' "$$sha" > "$$stage/AAC_GIT_SHA"; \
	mv -T "$$stage" "$$rel"; \
	got=$$(cd $$rel && $(R2_REL_ENV) $(R2_RPY) -c 'import aac, pathlib; print(pathlib.Path(aac.__file__).resolve)'); \
	test "$$got" = "$$rel/src/aac/__init__.py" || { echo "r2-release: aac imports from $$got, not the release" >&2; exit 1; }; \
	echo "r2-release: $$rel"

ev-rd: r2-release
	@set -e; trap 'rm -rf $(R2_REL)' EXIT; \
	old=$$($(R2_PY) -c "$$R2_FIND" runs EV ev-rd-pooled 0); \
	if test -n "$$old"; then echo "ev-rd: reusing sealed $$old"; exit 0; fi; \
	cd $(R2_REL) && $(R2_REL_ENV) $(R2_RPY) -m aac.components.c2_evidence_run run --config $(R2_EVRD_CONFIG) --seed 0

r2-runs: r2-release
	@set -e; trap 'rm -rf $(R2_REL)' EXIT; \
	for cfg in $(R2_CONFIGS); do name=$$(basename $$cfg .yaml); for s in $(R2_SEEDS); do \
 rid=$$($(R2_PY) -c "$$R2_FIND" runs R2 $$name $$s); \
 if test -n "$$rid"; then echo "r2-runs: reusing sealed $$rid ($$name seed $$s)" >&2; \
 else \
 rid=$$(cd $(R2_REL) && $(R2_REL_ENV) $(R2_AAC) run-id --config $$cfg --seed $$s); \
 (cd $(R2_REL) && $(R2_REL_ENV) $(R2_AAC) run --config $$cfg --seed $$s) >&2; \
 fi; \
 (cd $(R2_REL) && $(R2_REL_ENV) $(R2_AAC) validate --run $$rid) >&2; \
 $(R2_PY) -c "$$R2_LINE" runs/$$rid "r2-runs: $$name seed $$s"; \
	done; done

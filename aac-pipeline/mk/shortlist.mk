TW_WORLD = build/tinyworld/world
SL_WORKERS ?= 8
SL_PY = $(VENV)/bin/python
SL_THREADS_ENV = CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SL_SMOKE_SENTINELS = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0
SL_SMOKE_CONFIG = configs/tinyworld/sl-tw.yaml

.PHONY: tinyworld shortlist shortlist-smoke

tinyworld:
	$(SL_PY) -m aac.data.tinyworld make --out $(TW_WORLD)

shortlist:
	$(SL_THREADS_ENV) nice -n 10 $(SL_PY) -m aac.data.shortlist run --config $(RUN) --seed $(SEED) --workers $(SL_WORKERS)

shortlist-smoke: tinyworld
	@set -e; \
	one=$$($(SL_SMOKE_SENTINELS) AAC_OUT=build/tw-sl $(SL_THREADS_ENV) $(SL_PY) -m aac.data.shortlist run \
 --config $(SL_SMOKE_CONFIG) --seed 0 --workers 2 --force | sed -n 's/^shortlist: run_id=\(SL-[0-9a-f]*-s[0-9]*\) shortlist_sha256=\([0-9a-f]*\)$$/\1 \2/p'); \
	test -n "$$one" || { echo "shortlist-smoke: first run printed no run_id/shortlist_sha256 line" >&2; exit 1; }; \
	echo "shortlist-smoke: $$one"; \
	two=$$($(SL_SMOKE_SENTINELS) AAC_OUT=build/tw-sl2 $(SL_THREADS_ENV) $(SL_PY) -m aac.data.shortlist run \
 --config $(SL_SMOKE_CONFIG) --seed 0 --workers 1 --force | sed -n 's/^shortlist: run_id=\(SL-[0-9a-f]*-s[0-9]*\) shortlist_sha256=\([0-9a-f]*\)$$/\1 \2/p'); \
	test "$$one" = "$$two" || { echo "shortlist-smoke: FAIL: build/tw-sl gave '$$one', build/tw-sl2 gave '$$two'" >&2; exit 1; }; \
	echo "shortlist-smoke: build/tw-sl and build/tw-sl2 agree (2 workers vs 1 worker)"

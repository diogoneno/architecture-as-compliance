.PHONY: bodega-smoke bodega-score cpu-host-bodega

BODEGA_TW = configs/tinyworld/bodega-tw.yaml
BODEGA_TW_OUT = build/tw-pilot
BODEGA_TW_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=$(BODEGA_TW_OUT)
BODEGA_CHECK_PY = import sys; from aac.attacks.bodega import text_free_check; from aac.run.paths import root; \
 print(text_free_check(root, sys.argv[1]))

bodega-smoke: pilot-smoke
	@set -e; \
	$(BODEGA_TW_ENV) CUDA_VISIBLE_DEVICES= $(AAC) bodega-score --config $(BODEGA_TW) --seed 0 --force >&2; \
	rid=$$($(BODEGA_TW_ENV) $(AAC) run-id --config $(BODEGA_TW) --seed 0); \
	n=$$($(BODEGA_TW_ENV) $(VENV)/bin/python -c '$(BODEGA_CHECK_PY)' $$rid); \
	$(BODEGA_TW_ENV) $(AAC) reports build --table 4.4 --allow-pending >&2; \
	echo "bodega-smoke: $$rid rows=$$n text-free ok"

bodega-score:
	@test -n "$(CONFIG)" || { echo "bodega-score: set CONFIG=<aac-bodega/1 config> [SEED=0] [FORCE=1]" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES= $(AAC) bodega-score --config $(CONFIG) --seed $(SEED) $(if $(FORCE),--force)

cpu-host-bodega: export BENGIO_ID = $(BENGIO_ID_PY)
cpu-host-bodega: cpu-host-release
	@set -e; trap 'rm -rf $(BENGIO_REL); rmdir $(CURDIR)/build/cpu-host-release 2>/dev/null || true' EXIT; \
	test -n "$(CONFIG)" || { echo "cpu-host-bodega: set CONFIG=<aac-bodega/1 config> [SEED=0]" >&2; exit 1; }; \
	test -f "$(BENGIO_REL)/$(CONFIG)" || { echo "cpu-host-bodega: $(CONFIG) is not in the release $(BENGIO_SHA) (commit it first)" >&2; exit 1; }; \
	id=$$(cd $(BENGIO_REL) && $(BENGIO_REL_ENV) $(BENGIO_PY) -c "$$BENGIO_ID" $(CONFIG)) || exit 1; set -- $$id; \
	test "$$1" = BS || { echo "cpu-host-bodega: $(CONFIG) is regime $$1, not BS" >&2; exit 1; }; \
	echo "cpu-host-bodega: release $(BENGIO_REL)"; \
	if test -n "$(DRY)"; then \
 rid=$$(cd $(BENGIO_REL) && $(BENGIO_REL_ENV) $(BENGIO_AAC) run-id --config $(CONFIG) --seed $(SEED)); \
 echo "cpu-host-bodega: $$2 seed $(SEED) run_id=$$rid (dry run, nothing scored)"; exit 0; \
	fi; \
	if pgrep -u "$$(id -u)" -f 'aac[.]serve[.]victim_server' >/dev/null 2>&1; then \
 echo "cpu-host-bodega: an R3 victim server runs on this host; never overlap an attack session (operational constraint)" >&2; exit 1; fi; \
	cd $(BENGIO_REL) && $(BENGIO_REL_ENV) $(BENGIO_AAC) bodega-score --config $(CONFIG) --seed $(SEED) $(if $(FORCE),--force)

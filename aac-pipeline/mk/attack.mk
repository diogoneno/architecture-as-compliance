.PHONY: attack-assets runner-test runner-test-full attack attack-session-smoke pilot-smoke projection \
 attack-staging-clean

ifeq ($(origin HF_HOME),undefined)
ATK_HF_ENV = HF_HOME=$(CURDIR)/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
ATK_HF_ENV = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
endif
ATK_SMOKE_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=build/tw-attack
ATK_SMOKE_CONFIG = configs/attack/serve-smoke.yaml
ATTACK_SESSION ?= scripts/attack_session.sh

attack-assets:
	$(VENV)/bin/python -m aac.attacks.assets

runner-test: attack-assets
	scripts/attack_env.sh .venv-attack/bin/python -m unittest discover -s attacks/runner/tests -t . -v

runner-test-full: runner-test
	scripts/attack_env.sh .venv-attack/bin/python -m unittest attacks.runner.tests.full_build_check -v

ATK_CFG_PY = import sys; from aac.run.config import load_raw; a = load_raw(sys.argv[1]).get("attack") or {}; \
 t = a.get("session_timeout_s"); print(a.get("runner_device") or "cpu", t if t else "-")

attack:
	@set -e; cfg=$$($(VENV)/bin/python -c '$(ATK_CFG_PY)' $(RUN)) || exit 2; set -- $$cfg; dev=$$1; \
	to=$(if $(ATTACK_TIMEOUT),$(ATTACK_TIMEOUT),$$2); \
	case "$$dev" in cpu) gpu= ;; cuda) gpu=$${AAC_GPU:-0} ;; \
 *) echo "attack: runner_device $$dev is not cpu or cuda" >&2; exit 2 ;; esac; \
	if [ "$$to" = - ]; then set --; else set -- --timeout "$$to"; fi; \
	$(ATK_HF_ENV) AAC_GPU=$$gpu $(ATTACK_SESSION) $(RUN) $(SEED) --device "$$dev" "$$@" $(if $(ATTACK_FORCE),--force)

attack-session-smoke:
	$(ATK_SMOKE_ENV) $(AAC) run --config configs/regimes/r1-chrono-full.yaml --seed 0 --force
	$(ATK_SMOKE_ENV) $(MAKE) --no-print-directory attack RUN=$(ATK_SMOKE_CONFIG) SEED=0 ATTACK_FORCE=1
	@rid=$$($(ATK_SMOKE_ENV) $(AAC) run-id --config $(ATK_SMOKE_CONFIG) --seed 0) && \
	$(ATK_SMOKE_ENV) $(AAC) validate --run $$rid >/dev/null && \
	$(ATK_SMOKE_ENV) $(AAC) ledger-check build/tw-attack/runs/$$rid >/dev/null && \
	echo "attack-session-smoke: $$rid L2 ok"

PILOT_OUT = build/tw-pilot
PILOT_ENV = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=$(PILOT_OUT)
PILOT_CONFIG = configs/tinyworld/r3-tw-pilot.yaml

pilot-smoke: tinyworld
	@set -e; \
	rid=$$($(PILOT_ENV) $(AAC) run-id --config $(FC_SL) --seed 0); \
	if test -f $(PILOT_OUT)/runs/$$rid/SEALED; then echo "pilot-smoke: reusing sealed $$rid" >&2; \
	else $(PILOT_ENV) $(FC_CPU_ENV) $(FC_PY) -m aac.data.shortlist run --config $(FC_SL) --seed 0 --workers 2 >&2; fi; \
	rid=$$($(PILOT_ENV) $(AAC) run-id --config $(FC_EV) --seed 0); \
	if test -f $(PILOT_OUT)/runs/$$rid/SEALED; then echo "pilot-smoke: reusing sealed $$rid" >&2; \
	else $(PILOT_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(FC_PY) -m aac.components.c2_evidence_run run --config $(FC_EV) \
 --seed 0 >&2; fi; \
	for cfg in $(FC_C1) $(FC_C2S); do \
 rid=$$($(PILOT_ENV) $(AAC) run-id --config $$cfg --seed 0); \
 if test -f $(PILOT_OUT)/runs/$$rid/SEALED; then echo "pilot-smoke: reusing sealed $$rid" >&2; \
 else $(PILOT_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) train --config $$cfg --seed 0 >&2; fi; \
	done; \
	rid=$$($(PILOT_ENV) $(AAC) run-id --config $(FC_R1) --seed 0); \
	if test -f $(PILOT_OUT)/runs/$$rid/SEALED; then echo "pilot-smoke: reusing sealed $$rid" >&2; \
	else $(PILOT_ENV) CUDA_VISIBLE_DEVICES= $(FC_HF_ENV) $(AAC) run --config $(FC_R1) --seed 0 >&2; fi; \
	$(PILOT_ENV) CUDA_VISIBLE_DEVICES= $(MAKE) --no-print-directory attack RUN=$(PILOT_CONFIG) SEED=0 ATTACK_FORCE=1; \
	rid=$$($(PILOT_ENV) $(AAC) run-id --config $(PILOT_CONFIG) --seed 0); \
	$(PILOT_ENV) $(AAC) validate --run $$rid >/dev/null; \
	$(PILOT_ENV) $(AAC) ledger-check $(PILOT_OUT)/runs/$$rid >/dev/null; \
	echo "pilot-smoke: $$rid L2 ok"; \
	$(PILOT_ENV) $(VENV)/bin/python -m aac.attacks.projection --run-id $$rid \
 --out $(PILOT_OUT)/reports/r3-budget-projection.csv >&2; \
	n=$$($(VENV)/bin/python -m aac.attacks.projection --check $(PILOT_OUT)/reports/r3-budget-projection.csv \
 | sed -n 's/^projection-check: \([0-9]*\) rows ok$$/\1/p'); \
	test -n "$$n"; echo "pilot-projection: $$n rows ok"

projection:
	@test -n "$(RUN_ID)" || { echo "projection: set RUN_ID=<R3 run id>" >&2; exit 2; }
	$(VENV)/bin/python -m aac.attacks.projection --run-id $(RUN_ID) $(if $(OUT),--out $(OUT))
	$(VENV)/bin/python -m aac.attacks.projection --check $(if $(OUT),$(OUT),$$($(VENV)/bin/python -c \
 'from aac.run.paths import reports_dir; print(reports_dir / "r3-budget-projection.csv")'))

attack-staging-clean:
	@d=$${TMPDIR:-/tmp}; \
	list=$$(find "$$d" -mindepth 1 -maxdepth 1 -type d -name 'aac-attack-*' -user "$$(id -u)" -mmin +2880 \
 2>/dev/null | LC_ALL=C sort); \
	n=$$(printf '%s' "$$list" | grep -c . || true); \
	echo "attack-staging-clean: $$n staging directories older than 2 days under $$d"; \
	[ -z "$$list" ] || printf '%s\n' "$$list"; \
	if [ "$(CONFIRM)" = 1 ] && [ -n "$$list" ]; then \
 printf '%s\n' "$$list" | while IFS= read -r p; do rm -rf -- "$$p"; done; \
 echo "attack-staging-clean: removed $$n"; \
	elif [ -n "$$list" ]; then echo "attack-staging-clean: nothing removed (CONFIRM=1 removes them)"; fi

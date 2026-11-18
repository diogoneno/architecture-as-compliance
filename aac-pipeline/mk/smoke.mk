.PHONY: lint-determinism smoke-determinism

lint-determinism:
	$(AAC) lint-determinism

smoke-determinism:
	$(PYTEST) tests/determinism/test_t_d1.py -q -p no:cacheprovider

SMOKE_SENTINELS = AAC_GIT_SHA=0000000000000000000000000000000000000000 AAC_RUN_EPOCH=0 HF_HUB_OFFLINE=1 AAC_OUT=build/smoke
OUT = build/smoke
SMOKE_STEPS = fixtures mini-runs validate-all reports evidence-pack verify-pack lint-imports lint-determinism golden-diff
GOLDEN = tests/golden

.PHONY: smoke smoke-run validate-all golden-diff bless $(addprefix smoke-step-,$(SMOKE_STEPS))

smoke:
	@exec 9>.smoke.lock; flock -n 9 || { echo "smoke already running (aac-pipeline/.smoke.lock held)"; exit 1; }; \
	$(SMOKE_SENTINELS) $(MAKE) --no-print-directory smoke-run

smoke-run:
	@for step in $(SMOKE_STEPS); do \
 if $(MAKE) --no-print-directory smoke-step-$$step; then echo "SMOKE $$step: ok$$(cat .smoke-note 2>/dev/null)"; rm -f .smoke-note; \
 else rm -f .smoke-note; echo "SMOKE $$step: FAIL"; exit 1; fi; \
	done

smoke-step-fixtures:
	$(AAC) fixtures build
	$(AAC) index docs --from-index runs/INDEX.json
smoke-step-mini-runs:
	@$(MAKE) --no-print-directory mini-runs
	$(AAC) index manifest
smoke-step-validate-all: validate-all
smoke-step-reports:
	@test -f $(OUT)/reports/table-4.7-verifiability.csv || { \
 echo "reports: priming $(OUT)/reports/table-4.7-verifiability.csv (fresh namespace; Table 4.8 RQ4 reads the previous verification)"; \
 $(MAKE) --no-print-directory reports >/dev/null && $(AAC) pack build >/dev/null && { scripts/regen_check.sh --against-committed >/dev/null || true; }; \
 test -f $(OUT)/reports/table-4.7-verifiability.csv || { echo "reports: priming did not write table-4.7"; exit 1; }; }
	@$(MAKE) --no-print-directory reports
smoke-step-evidence-pack:
	$(AAC) pack build
smoke-step-verify-pack:
	scripts/regen_check.sh --against-committed
smoke-step-lint-imports:
	@$(MAKE) --no-print-directory lint-imports
smoke-step-lint-determinism: lint-determinism
smoke-step-golden-diff:
	@$(MAKE) --no-print-directory golden-diff && printf ' (day1_docs -v1 %s)' "$$(git rev-parse '-v1^{commit}' | cut -c1-12)" > .smoke-note

validate-all:
	@n=0; for rid in $$($(AAC) matrix list --matrix $(MATRIX) | cut -d' ' -f3); do \
 $(AAC) validate --run $$rid > /dev/null || { echo "validate-all: $$rid failed"; exit 1; }; n=$$((n+1)); \
	done; echo "validate-all: $$n sealed runs, 0 violations, sealed counts agree"

golden-diff:
	@test -d $(GOLDEN)/reports && test -d $(GOLDEN)/evidence-pack && test -f $(GOLDEN)/INDEX.json \
 || { echo "golden-diff: tests/golden/ incomplete (a missing golden fails, never skips)"; exit 1; }
	diff -r $(OUT)/reports $(GOLDEN)/reports
	diff -r --exclude=decision-log $(OUT)/evidence-pack $(GOLDEN)/evidence-pack
	diff $(OUT)/runs/INDEX.json $(GOLDEN)/INDEX.json
	@want=$$(git rev-parse '-v1^{commit}') && have=$$(jq -r .day1_docs._v1_commit $(OUT)/runs/INDEX.json) \
 && test "$$have" = "$$want" || { echo "golden-diff: $(OUT)/runs/INDEX.json day1_docs._v1_commit '$$have' != -v1^{commit} '$$want'"; exit 1; }

bless:
	scripts/bless.sh

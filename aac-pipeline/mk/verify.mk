COMMIT ?=
WITH_REPORTS ?=
VERIFY_OUT ?=
PACK ?=

.PHONY: verify-clean-checkout integrity-selftest

verify-clean-checkout:
	@test -n "$(COMMIT)" || { echo "verify-clean-checkout: give COMMIT=<commit>"; exit 2; }
	scripts/verify_clean_checkout.sh "$(COMMIT)" $(if $(filter 1,$(WITH_REPORTS)),--with-reports) $(if $(VERIFY_OUT),--out "$(VERIFY_OUT)")

integrity-selftest:
	scripts/integrity_selftest.sh $(if $(PACK),"$(PACK)")

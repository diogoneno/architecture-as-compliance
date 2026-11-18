NS ?= build/pack-tracer
RUNS ?= thin
KEEP ?=
PACK_ENV_A = SOURCE_DATE_EPOCH=0 TZ=UTC PYTHONHASHSEED=0 AAC_LISTING=sorted
PACK_ENV_B = SOURCE_DATE_EPOCH=1700000000 TZ=Pacific/Kiritimati PYTHONHASHSEED=42 AAC_LISTING=shuffled

.PHONY: pack-namespace pack-namespace-clean pack-tracer analyst-demo

pack-namespace:
	env -u AAC_OUT $(VENV)/bin/python -m aac.pack.namespace link --out $(NS) --runs $(RUNS)
	env -u AAC_OUT $(AAC) reports build --table D.1 --out $(CURDIR)/$(NS)/reports
	env -u AAC_OUT $(AAC) reports build --table D.2 --out $(CURDIR)/$(NS)/reports

pack-namespace-clean:
	env -u AAC_OUT $(VENV)/bin/python -m aac.pack.namespace clean --out $(NS)

pack-tracer: SHELL := /bin/bash
pack-tracer:
	@set -u; ns="$(CURDIR)/$(NS)"; aac="$(CURDIR)/$(AAC)"; rc=0; reason=""; \
	finish { \
 if [ -z "$(KEEP)" ]; then $(MAKE) --no-print-directory pack-namespace-clean NS=$(NS) | tail -1; fi; \
 if [ -n "$$reason" ]; then echo "pack-tracer: $(RUNS) determinism DIFFERS ($$reason)"; exit 1; fi; \
 echo "$$line"; exit 0; }; \
	if [ -e "$$ns" ]; then echo "pack-tracer: $(RUNS) determinism DIFFERS ($(NS) exists; make pack-namespace-clean NS=$(NS))"; exit 1; fi; \
	$(MAKE) --no-print-directory pack-namespace NS=$(NS) RUNS=$(RUNS) > /dev/null || { reason="namespace build failed"; finish; }; \
	echo "pack-tracer: load average at start $$(cut -d' ' -f1-3 /proc/loadavg); AAC_PACK_WORKERS=$${AAC_PACK_WORKERS:-default}"; \
	( cd / && s=$$SECONDS; env AAC_ROOT="$(CURDIR)" AAC_OUT="$$ns" $(PACK_ENV_A) "$$aac" pack build --out "$$ns/pack-a" > "$$ns/build-a.log" 2>&1; \
 r=$$?; echo $$((SECONDS - s)) > "$$ns/seconds-a"; exit $$r ) & pa=$$!; \
	( cd "$$HOME" && s=$$SECONDS; env AAC_ROOT="$(CURDIR)" AAC_OUT="$$ns" $(PACK_ENV_B) "$$aac" pack build --out "$$ns/pack-b" > "$$ns/build-b.log" 2>&1; \
 r=$$?; echo $$((SECONDS - s)) > "$$ns/seconds-b"; exit $$r ) & pb=$$!; \
	wait $$pa || { tail -5 "$$ns/build-a.log" >&2; reason="build a failed"; }; \
	wait $$pb || { tail -5 "$$ns/build-b.log" >&2; [ -n "$$reason" ] || reason="build b failed"; }; \
	[ -z "$$reason" ] || finish; \
	grep -E '^(pack:|runs without decision-log:)' "$$ns/build-a.log" | cut -c1-400; \
	diff -r "$$ns/pack-a" "$$ns/pack-b" > /dev/null || { diff -r "$$ns/pack-a" "$$ns/pack-b" | head -10 >&2; reason="builds differ"; }; \
	( cd "$$ns/pack-a" && LC_ALL=C sha256sum -c --quiet MANIFEST.sha256 ) || { [ -n "$$reason" ] || reason="MANIFEST.sha256 check failed"; }; \
	leak { grep -rIn $$1 -e "$$2" "$$ns/pack-a" "$$ns/pack-b" 2>/dev/null | grep -v '^[^:]*:[0-9]*:# host_change: '; }; \
	for spec in "-F|$(CURDIR)" "-P|(?<![\w.-])$$(hostname)(?![\w-])" "-P|(?<![\w.-])$$USER(?![\w-])"; do \
 if leak "$${spec%%|*}" "$${spec#*|}" | grep -q .; then \
 leak "$${spec%%|*}" "$${spec#*|}" | cut -d: -f1 | sort -u | head -5 >&2; [ -n "$$reason" ] || reason="leak of '$${spec#*|}'"; fi; \
	done; \
	sv="$$ns/pack-a/schema-validation.txt"; v { sed -n "s/^$$1=//p" "$$sv"; }; \
	line="pack-tracer: $(RUNS) determinism identical records=$$(v records) violations=$$(v violations) claim_text=$$(v records_with_claim_text) user_id=$$(v records_with_user_id) seconds=$$(cat $$ns/seconds-a)/$$(cat $$ns/seconds-b)"; \
	finish

DEMO_CASE_REAL = configs/analyst/demo-case.real.json
REAL_EVENTS = analyst/analyst-events.jsonl
analyst-demo: NS = build/analyst-demo
analyst-demo: SHELL := /bin/bash
analyst-demo:
	@set -u; ns="$(CURDIR)/$(NS)"; py="$(CURDIR)/$(VENV)/bin/python"; reason=""; line=""; \
	finish { \
 if [ -z "$(KEEP)" ]; then $(MAKE) --no-print-directory pack-namespace-clean NS=$(NS) | tail -1; fi; \
 if [ -n "$$reason" ]; then echo "analyst-demo: FAILED ($$reason)"; exit 1; fi; \
 echo "$$line"; exit 0; }; \
	case "$(NS)" in build/?*) ;; *) echo "analyst-demo: FAILED (NS=$(NS) is not under build/)"; exit 1;; esac; \
	if [ -e "$$ns" ]; then echo "analyst-demo: FAILED ($(NS) exists; make pack-namespace-clean NS=$(NS))"; exit 1; fi; \
	real_sum { if [ -e "$(REAL_EVENTS)" ]; then sha256sum < "$(REAL_EVENTS)"; else echo absent; fi; }; \
	before=$$(real_sum); \
	rid=$$("$$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record_id"])' "$(DEMO_CASE_REAL)"); \
	src=$$("$$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_run_id"])' "$(DEMO_CASE_REAL)"); \
	$(MAKE) --no-print-directory pack-namespace NS=$(NS) RUNS=thin > /dev/null || { reason="namespace build failed"; finish; }; \
	env AAC_OUT="$$ns" $(CURDIR)/$(AAC) analyst record --record-id "$$rid" --action return || { reason="aac analyst record failed"; finish; }; \
	env AAC_OUT="$$ns" "$$py" -m aac.pack.analyst_join render --out "$$ns/join" || { reason="join failed"; finish; }; \
	line=$$("$$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); j=d["joined"]; assert d["n_events"]==1 and len(j)==1 and j[0]["record_id"]==sys.argv[2] and j[0]["run_id"]==sys.argv[3] and d["seals_reverified"]==[sys.argv[3]], "join row"; print("analyst-demo: 1 event joined record=%s run=%s seq=%s seals unchanged" % (j[0]["record_id"], j[0]["run_id"], j[0]["seq"]))' "$$ns/join/analyst-join.json" "$$rid" "$$src") || { reason="join file check failed"; finish; }; \
	[ "$$(real_sum)" = "$$before" ] || { reason="the real $(REAL_EVENTS) changed"; finish; }; \
	finish

KS_LIMIT ?=
KS_OUT ?= $(if $(KS_LIMIT),data/derived/averitec/ks-audit-limit-$(KS_LIMIT),data/derived/averitec/ks-audit)
KS_WORKERS ?= 8
KS_PHASE ?= A-ks
KS_PY = $(VENV)/bin/python

.PHONY: ks-domains ks-audit ks-pull

ks-domains:
	$(KS_PY) -m aac.data.kstore domains --out configs/factchecker-domains.txt

ks-audit:
	$(KS_PY) -m aac.data.kstore audit --out-dir $(KS_OUT) $(if $(KS_LIMIT),--limit $(KS_LIMIT)) --workers $(KS_WORKERS)

ks-pull:
	@set -e; \
	case "$(KS_OUT)" in data/derived/*) ;; *) echo "ks-pull: KS_OUT must lie under data/derived/" >&2; exit 1 ;; esac; \
	echo "$(KS_PHASE)" | grep -Eq '^[A-Za-z0-9_-]+$$' || { echo "ks-pull: bad KS_PHASE" >&2; exit 1; }; \
	eval "$$($(AAC) dispatch env)"; \
	SSH="ssh $$AAC_SSH_OPTS $$AAC_REMOTE_HOST"; \
	src="$$AAC_DERIVED/$(patsubst data/derived/%,%,$(KS_OUT))"; \
	tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	for f in ks-audit.csv ks-stats.json ks-audit-timing.json; do \
 rsync -t --checksum -e "ssh $$AAC_SSH_OPTS" "$$AAC_REMOTE_HOST:$$src/$$f" "$$tmp/$$f"; \
	done; \
	$$SSH "cd '$$src' && sha256sum ks-audit.csv ks-stats.json ks-audit-timing.json" < /dev/null > "$$tmp/remote.sha256"; \
	(cd "$$tmp" && sha256sum --strict -c remote.sha256); \
	sha=$$($(KS_PY) -c 'import json, sys; print(json.load(open(sys.argv[1]))["release_sha"])' "$$tmp/ks-audit-timing.json"); \
	echo "$$sha" | grep -Eq '^[0-9a-f]{40}$$' || { echo "ks-pull: release_sha '$$sha' is not a 40-hex commit" >&2; exit 1; }; \
	log=$$($$SSH "ls -1 $$AAC_RELEASES/$$sha/logs/$(KS_PHASE)-*.log 2>/dev/null | tail -n 1" < /dev/null); \
	test -n "$$log" || { echo "ks-pull: no logs/$(KS_PHASE)-*.log in release $$sha" >&2; exit 1; }; \
	logsha=$$($$SSH "sha256sum '$$log'" < /dev/null | cut -d' ' -f1); \
	echo "ks-pull: release $$sha log $$(basename "$$log") sha256=$$logsha"; \
	$(KS_PY) -m aac.data.kstore place --from "$$tmp" --log "$$log" --log-sha256 "$$logsha"

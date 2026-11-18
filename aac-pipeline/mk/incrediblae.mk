INC_PY = $(VENV)/bin/python

.PHONY: incrediblae-ingest

incrediblae-ingest:
	@if [ -e AAC_GIT_SHA ]; then echo "incrediblae-ingest: FAIL: this is a gpu-host release (AAC_GIT_SHA present); run it on cpu-host" >&2; exit 1; fi
	$(INC_PY) -m aac.data.incrediblae

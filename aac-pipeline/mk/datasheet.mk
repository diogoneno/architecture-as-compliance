DATASHEET_MANIFEST ?= data/derived/MANIFEST.sha256
REAL_ENV = env -u AAC_OUT

.PHONY: data-manifest datasheet datasheet-gate

data-manifest:
	$(REAL_ENV) $(VENV)/bin/python -m aac.data.manifest write
	$(REAL_ENV) $(VENV)/bin/python -m aac.data.manifest check

datasheet:
	$(REAL_ENV) $(VENV)/bin/python -m aac.pack.datasheet render --manifest $(DATASHEET_MANIFEST)

datasheet-gate:
	@$(REAL_ENV) $(VENV)/bin/python -m aac.data.manifest gate

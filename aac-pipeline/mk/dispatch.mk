PHASE ?= A
SEEDS ?= 0
RUN_ID ?=
BATCH ?=
DEST ?=

.PHONY: gpu-check gpu-push gpu-run gpu-status gpu-pull probe install-host-tools backup sync-from-gpu-host install-cron downloads-start downloads-status

gpu-check:
	scripts/gpu_check.sh

gpu-push:
	SHA=$(SHA) scripts/gpu_push.sh

gpu-run:
	SHA=$(SHA) scripts/gpu_run.sh $(PHASE) $(if $(BATCH),--batch $(BATCH),$(RUN) $(SEEDS))

gpu-status: export AAC_STATUS_PHASE := $(PHASE)
gpu-status: export AAC_STATUS_SHA := $(SHA)
gpu-status:
	SHA="$$AAC_STATUS_SHA" scripts/gpu_status.sh "$$AAC_STATUS_PHASE"

gpu-pull:
	@test -n "$(RUN_ID)" || { echo "usage: make gpu-pull RUN_ID=<run id>"; exit 1; }
	scripts/gpu_pull.sh $(RUN_ID)

probe:
	$(AAC) dispatch probe --host gpu-host
	$(AAC) dispatch probe --host cpu-host

install-host-tools:
	scripts/install_host_tools.sh

backup:
	$(if $(DEST),DEST=$(DEST)) scripts/backup.sh

sync-from-gpu-host:
	scripts/sync_from_turing.sh

install-cron:
	scripts/install_cron.sh

downloads-start:
	scripts/downloads_start.sh

downloads-status:
	scripts/downloads_start.sh --status

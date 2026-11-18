ATTACK_VENV = .venv-attack
ATTACK_PY = $(ATTACK_VENV)/bin/python
ATTACK_REQ = attacks/requirements-attack.txt
ATTACK_LOCK ?= attacks/attack-cpu.lock.txt
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
ATTACK_INDEXES = --index-url https://pypi.org/simple --extra-index-url $(TORCH_INDEX) --index-strategy unsafe-best-match

.PHONY: env-attack env-attack-lock env-gate

env-attack-lock:
	cd attacks && $(UV) pip compile $(notdir $(ATTACK_REQ)) --generate-hashes --python-version 3.10 $(ATTACK_INDEXES) -o $(notdir $(ATTACK_LOCK))

env-attack:
	rm -rf $(ATTACK_VENV)
	$(UV) venv --python 3.10 $(ATTACK_VENV)
	grep -vE '^[A-Za-z0-9_.-]+ @ git\+' $(ATTACK_LOCK) > $(ATTACK_VENV)/lock-pypi.txt
	grep -E '^[A-Za-z0-9_.-]+ @ git\+' $(ATTACK_LOCK) > $(ATTACK_VENV)/lock-vcs.txt
	$(UV) pip sync --python $(ATTACK_PY) --require-hashes $(ATTACK_INDEXES) $(ATTACK_VENV)/lock-pypi.txt
	$(UV) pip install --python $(ATTACK_PY) --no-deps $(ATTACK_INDEXES) -r $(ATTACK_VENV)/lock-vcs.txt
	$(UV) pip uninstall --python $(ATTACK_PY) tensorflow tensorflow-cpu
	awk '/^tensorflow-cpu==/{p=1;print;next} p&&/^[^ ]/{p=0} p&&!/^ *#/{print}' $(ATTACK_LOCK) > $(ATTACK_VENV)/tensorflow-cpu.lock-block.txt
	$(UV) pip install --python $(ATTACK_PY) --no-deps --require-hashes $(ATTACK_INDEXES) -r $(ATTACK_VENV)/tensorflow-cpu.lock-block.txt
	$(UV) pip freeze --python $(ATTACK_PY) > $(ATTACK_VENV)/freeze.txt
	scripts/attack_env.sh python -c "import sys, transformers, OpenAttack, tensorflow; print('env-attack ok: python', sys.version.split[0], 'transformers', transformers.__version__, 'tensorflow', tensorflow.__version__)"

env-gate:
	scripts/env_gate.sh

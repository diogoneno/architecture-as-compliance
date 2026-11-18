T_L2_STATE ?= configs/architectural isolation constraint-enforcement.yaml

.PHONY: lint-imports

lint-imports:
	cd $(CURDIR) && scripts/t_l2_gate.sh $(T_L2_STATE)

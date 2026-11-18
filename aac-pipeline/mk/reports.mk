.PHONY: reports report reports-index

reports:
	$(AAC) reports build --all
	$(AAC) reports index

report:
	$(AAC) reports build --table $(TABLE)

reports-index:
	$(AAC) reports index

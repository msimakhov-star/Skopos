.PHONY: dev check
dev:
	./run.sh --reload
check:
	./.venv/bin/python -m skopos.selfcheck

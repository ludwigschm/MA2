.PHONY: e2e

e2e:
	python tools/e2e_smoke.py $(ARGS)

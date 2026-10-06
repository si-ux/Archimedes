.PHONY: demo console test lint train train-smoke filter-smoke

demo:            ## browser demo - open the forwarded port 8000
	python -m uvicorn webdemo.server:app --host 0.0.0.0 --port 8000

console:         ## Python console without the browser (scripts: python -m archimedes_fe.scripting file.py)
	python -m archimedes_fe.scripting

test:
	python -m pytest -q

lint:
	ruff check .

train:           ## train on clips you recorded in the demo (+ HaGRID if extracted)
	python scripts/train_static.py --clips data/clips $(if $(wildcard data/hagrid_lm.npz),--hagrid data/hagrid_lm.npz)

train-smoke:     ## check the training pipeline end to end on synthetic hands
	python scripts/train_static.py --synthetic --clips /nonexistent --out /tmp/smoke.joblib --report /tmp/smoke.md

filter-smoke:
	python scripts/evaluate_filter.py --synthetic

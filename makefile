
help:
	@echo Script to automate parts for this projects.

clean: clean_cache clean_runs

clean_cache: 
	rm -r ./cache/*

clean_runs: 
	rm -r ./runs/*

train:
	accelerate launch accelerate_llm_khubist.py

preprocess:
	python prepreprocessing_khubist.py

# Added pytest, linter

vega_nodes:
	srun -p gpu --gres=gpu:4 --nodes=1 --time=10:00:00 --mem=48GB --cpus-per-gpu=4 --pty bash

vega_drop_nodes:
	scancel -u euerikl

ready_test:
	pip install -e .
	pip install -r requirements_dev.txt
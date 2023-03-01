
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
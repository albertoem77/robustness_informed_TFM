#!make
include .env
.PHONY: install-ivae run-kegg run-reactome run-random run-scoring-kegg run-scoring-reactome run-scoring-random
.ONESHELL:

SHELL := /bin/bash
FRACS=$$(LANG=en_US seq ${FRAC_START} ${FRAC_STEP} ${FRAC_STOP})
SEEDS=$$(LANG=en_US seq ${SEED_START} ${SEED_STEP} ${SEED_STOP})
PY_FILES := isrobust_TFG/*.py


all: | install-ivae run-kegg  run-reactome  run-random  run-scoring-kegg  run-scoring-reactome  run-scoring-random

install-ivae:
	pixi install

run-kegg: install-ivae 
	rm -rf ${RESULTS_FOLDER}/ivae_kegg
	mkdir -p ${RESULTS_FOLDER}/ivae_kegg/logs/
	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		pixi run --environment cuda  python notebooks/00-train.py ivae_kegg ${DEBUG} {} \
		">" ${RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{}.out \
		"2>" ${RESULTS_FOLDER}/ivae_kegg/logs/train_seed-{}.err \
		::: $(SEEDS)
        
run-reactome: install-ivae 
	rm -rf ${RESULTS_FOLDER}/ivae_reactome
	mkdir -p ${RESULTS_FOLDER}/ivae_reactome/logs
	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		pixi run --environment cuda python notebooks/00-train.py ivae_reactome ${DEBUG} {} \
		">" ${RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{}.out \
		"2>" ${RESULTS_FOLDER}/ivae_reactome/logs/train_seed-{}.err \
		::: $(SEEDS)

run-random: install-ivae 
	rm -rf $$(printf "${RESULTS_FOLDER}/ivae_random-%s " $(FRACS))
	mkdir -p $$(printf "${RESULTS_FOLDER}/ivae_random-%s/logs " $(FRACS))
	parallel -j${N_GPU} CUDA_VISIBLE_DEVICES='$$(({%} - 1))' \
		pixi run --environment cuda python notebooks/00-train.py ivae_random-{2} ${DEBUG} {2} {1} \
		">" ${RESULTS_FOLDER}/ivae_random-{2}/logs/train_seed-{1}.out \
		"2>" ${RESULTS_FOLDER}/ivae_random-{2}/logs/train_seed-{1}.err \
		::: $(SEEDS) \
		::: $(FRACS)



run-scoring-kegg:
	pixi run --environment cuda papermill notebooks/01-compute_scores.ipynb - \
		-p model_kind ivae_kegg \
		> ${RESULTS_FOLDER}/ivae_kegg/logs/scoring.out \
		2> ${RESULTS_FOLDER}/ivae_kegg/logs/scoring.err
        
run-scoring-reactome:
	pixi run --environment cuda papermill notebooks/01-compute_scores.ipynb - \
		-p model_kind ivae_reactome \
		> ${RESULTS_FOLDER}/ivae_reactome/logs/scoring.out \
		2> ${RESULTS_FOLDER}/ivae_reactome/logs/scoring.err

run-scoring-random:
	parallel -j${N_CPU} \
		pixi run --environment cuda papermill notebooks/01-compute_scores.ipynb - \
		-p model_kind ivae_random-{} -p frac {} \
		">" ${RESULTS_FOLDER}/ivae_random-{}/logs/scoring.out \
		"2>" ${RESULTS_FOLDER}/ivae_random-{}/logs/scoring.err \
		::: $(FRACS)
                       


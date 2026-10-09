# Lumine — common tasks.
#
# Everything routes through `python -m lumine`; this file exists so the workflow is legible
# without reading the CLI source.

PY ?= python3
TASK ?= combat_defeat_and_chest
STAGE ?= pretrain
REGIONS ?= mondstadt liyue

.PHONY: help install test test-live rates tasks providers probe preview play \
        scripted collect corpus train eval clean all

help:
	@echo "make install     install runtime dependencies"
	@echo "make test        unit tests, no network"
	@echo "make test-live   tests that call a real VLM (needs a key)"
	@echo "make tasks       list the evaluation task suite"
	@echo "make providers   list pluggable VLM backends and how they compare"
	@echo "make probe       prove the configured key actually accepts images"
	@echo "make rates       measure the 5 / 30 / 0.5 Hz design on this machine"
	@echo "make preview     render one sandbox frame per task to PNG"
	@echo "make scripted    play one task with no model at all"
	@echo "make play        play one task with the configured VLM"
	@echo "make collect     record behaviour-cloning data from the scripted teacher"
	@echo "make train       train the 30 Hz action head"
	@echo "make eval        run the suite, in-distribution and held out"
	@echo "make all         scripted -> collect -> corpus -> train -> eval"

install:
	$(PY) -m pip install --user -r requirements.txt

test:
	$(PY) -m tests.test_core
	-$(PY) -m tests.test_sim

test-live:
	$(PY) -m tests.test_brain_live

rates:
	$(PY) -m lumine rates

tasks:
	$(PY) -m lumine tasks

providers:
	$(PY) -m lumine providers

probe:
	$(PY) -m lumine probe

preview:
	$(PY) -m lumine preview --out artifacts/sim_previews

scripted:
	$(PY) -m lumine play --config configs/scripted.yaml --task $(TASK)

play:
	$(PY) -m lumine play --task $(TASK) --video artifacts/$(TASK).mp4

collect:
	$(PY) -m lumine collect --minutes 10 --stage $(STAGE)

corpus:
	$(PY) -m lumine corpus

train:
	$(PY) -m lumine train --stage $(STAGE)

eval:
	$(PY) -m lumine eval --regions $(REGIONS) --seeds 0 1

all: scripted collect corpus train eval

clean:
	rm -rf artifacts/eval artifacts/checkpoints artifacts/sim_previews
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

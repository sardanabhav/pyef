export PDM_MULTIRUN_VERSIONS ?= 3.10 3.11
.DEFAULT_GOAL := help
SHELL := bash
DUTY := $(if $(VIRTUAL_ENV),,pdm run) duty

args = $(foreach a,$($(subst -,_,$1)_args),$(if $(value $a),$a="$($a)"))
check_quality_args = files
docs_args = host port
release_args = version
test_args = match

BASIC_DUTIES = \
	changelog \
	check-api \
	check-dependencies \
	clean \
	cov \
	docs \
	docs-deploy \
	format \
	release

QUALITY_DUTIES = \
	check-quality \
	check-docs \
	check-types \
	test-not-slow \
	test

.PHONY: help
help:
	@$(DUTY) --list

.PHONY: lock
lock:
	@pdm lock -G:all

.PHONY: setup
setup:
	@bash scripts/setup.sh

.PHONY: check
check:
	@#check-quality
	@pdm multirun -fei `echo $(PDM_MULTIRUN_VERSIONS) | sed "s/ /,/g"` duty check-types check-docs
	@$(DUTY) check-dependencies check-api

.PHONY: $(BASIC_DUTIES)
$(BASIC_DUTIES):
	@$(DUTY) $@ $(call args,$@)

.PHONY: $(QUALITY_DUTIES)
$(QUALITY_DUTIES):
	@pdm multirun -fei `echo $(PDM_MULTIRUN_VERSIONS) | sed "s/ /,/g"` duty $@ $(call args,$@)

precommit: format test-not-slow check
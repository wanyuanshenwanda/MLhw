# ==== 配置 ====
PYTHON ?= python
DATA_OUTPUT := datas/all_projects_final_features.csv

# ==== 默认目标 ====
.DEFAULT_GOAL := help

# ==== 目标 ====

## help: 查看所有可用命令
.PHONY: help
help:
	@echo "  make integrate           # 整合数据"
	@echo "  make run MODE=regressor  # 只跑回归"
	@echo "  make run MODE=classifier # 只跑分类"
	@echo "  make run                 # 默认都跑"
	@echo "  make clean"

.PHONY: integrate
integrate:  ## integrate datas
	@if [ -f $(DATA_OUTPUT) ]; then \
		echo "data has already in : $(DATA_OUTPUT)"; \
	else \
		cd src/data_preprocessing && PYTHONPATH=$(PWD) $(PYTHON) data_integration.py; \
	fi

MODE ?= all
.PHONY: run
run:  ## MODE=[classifier,regressor,all] default is all
	@echo "Running main.py with MODE=$(MODE)"
	cd src && PYTHONPATH=$(PWD) $(PYTHON) main.py --mode $(MODE)

## clean: 清理缓存和临时文件
.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	@echo "clean over"

.PHONY: help install install-dev setup-data download-model train test lint format clean docker-build docker-run api streamlit

help:
	@echo "SkimLit - Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make setup-data       - Download and setup data"
	@echo "  make download-model   - Download pretrained model"
	@echo "  make train            - Train model"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linting"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean generated files"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make api              - Start FastAPI server"
	@echo "  make streamlit        - Start Streamlit app"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

setup-data:
	@echo "Downloading PubMed 20k RCT dataset..."
	git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git data/pubmed-rct
	@echo "Data downloaded successfully"

download-model:
	@echo "Downloading pretrained model..."
	mkdir -p models
	wget https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip
	unzip skimlit_tribrid_model.zip -d models/
	rm skimlit_tribrid_model.zip
	@echo "Model downloaded successfully"

train:
	python train.py --config configs/model_config.yaml --experiment-name $(name)

train-full:
	python train.py --config configs/model_config.yaml --full-dataset --evaluate

test:
	pytest tests/ -v --cov=src --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow" --cov=src

lint:
	flake8 src/ api/ tests/
	mypy src/ api/

format:
	black src/ api/ tests/
	isort src/ api/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf logs/ experiments/

docker-build:
	docker build -t skimlit:latest .

docker-run:
	docker run -p 8000:8000 skimlit:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run app.py

predict:
	python predict.py --model-path models/skimlit_tribrid_model --text "$(text)"

predict-file:
	python predict.py --model-path models/skimlit_tribrid_model --file $(file)

setup-monitoring:
	mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
	@echo "Monitoring directories created"

# Development helpers
jupyter:
	jupyter notebook notebooks/

tensorboard:
	tensorboard --logdir experiments/

# CI/CD helpers
ci-test:
	pytest tests/ -v -m "not slow" --cov=src --cov-report=xml

ci-lint:
	black --check src/ api/ tests/
	isort --check-only src/ api/ tests/
	flake8 src/ api/ tests/ --max-line-length=100
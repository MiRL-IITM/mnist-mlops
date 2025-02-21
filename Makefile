.PHONY: setup data train test serve

setup:
	pip install -r requirements.txt

data:
	python -m src.data.download_mnist
	python -m src.data.preprocess_data

train:
	python -m src.models.train_model

test:
	python -m src.tests.test_model

serve:
	python -m src.api.app

.PHONY: setup data train test serve

setup:
	pip install -r requirements.txt

data:
	python src/data/download_mnist.py
	python src/data/preprocess_data.py

train:
	python src/models/train_model.py

test:
	pytest src/tests/

serve:
	python src/api/app.py

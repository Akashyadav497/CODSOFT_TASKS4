# Sales Prediction Automation & Runner Commands

.PHONY: help install eda train evaluate predict test app clean

help:
	@echo "Available commands:"
	@echo "  make install   : Install required dependencies"
	@echo "  make eda       : Run exploratory data analysis and generate charts"
	@echo "  make train     : Train all machine learning models and tune best"
	@echo "  make evaluate  : Evaluate models on test split and generate metrics"
	@echo "  make predict   : Run sample prediction via CLI"
	@echo "  make test      : Execute pytest test suite"
	@echo "  make app       : Launch interactive Streamlit web dashboard"

install:
	pip install -r requirements.txt

eda:
	python -m src.eda

train:
	python -m src.train

evaluate:
	python -m src.evaluate

predict:
	python -m src.predict --tv 150 --radio 30 --newspaper 20

test:
	pytest tests/ -v

app:
	streamlit run app.py

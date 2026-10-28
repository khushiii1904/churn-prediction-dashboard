.PHONY: install train test api dashboard up down

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=src python -m churn.train

test:
	pytest -q

api:
	PYTHONPATH=src:. uvicorn api.main:app --reload --port 8000

dashboard:
	API_URL=http://localhost:8000 streamlit run dashboard/app.py

up:
	docker compose up --build

down:
	docker compose down

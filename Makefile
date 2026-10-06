.PHONY: setup fast full test lint docker-fast docker-full clean
setup:
	python -m pip install -r requirements.txt
	cp -n .env.example .env || true
	mkdir -p results data papers
fast:
	python experiments/run_all.py --fast
full:
	python experiments/run_all.py --full
test:
	python -m pytest tests -q
docker-fast:
	docker compose up --build benchmark
docker-full:
	docker compose --profile full up --build full
clean:
	rm -rf results/*.json results/*.csv results/*.png __pycache__ .pytest_cache

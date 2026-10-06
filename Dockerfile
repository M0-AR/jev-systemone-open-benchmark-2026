FROM python:3.11-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY experiments ./experiments
COPY tests ./tests
COPY Makefile .
ENV PYTHONPATH=/work
CMD ["python", "experiments/run_all.py", "--fast"]

FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY migrations ./migrations
COPY tests ./tests
RUN pip install --no-cache-dir '.[test]' && useradd --uid 10001 --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "regchain.api:app", "--host", "0.0.0.0", "--port", "8000"]

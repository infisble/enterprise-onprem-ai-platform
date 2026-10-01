# API + worker image (same code, different command) - modular monolith.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first for layer caching
COPY pyproject.toml ./
RUN mkdir -p apps services packages scripts evaluations \
    && pip install --upgrade pip \
    && python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'));print('\n'.join(d['project']['dependencies']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt pytest

COPY apps ./apps
COPY services ./services
COPY packages ./packages
COPY scripts ./scripts
COPY evaluations ./evaluations
COPY tests ./tests
COPY data ./data

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=10 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "python scripts/bootstrap_demo.py --if-empty && uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --workers 1"]

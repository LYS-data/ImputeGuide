FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/imputeguide
COPY requirements/core.txt requirements/core.txt
RUN python -m pip install --upgrade pip==24.0 \
    && python -m pip install -r requirements/core.txt

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY imputers/ imputers/
COPY utils/ utils/
COPY external_dependencies/ external_dependencies/
COPY DiffPuter-main/ DiffPuter-main/
COPY configs/ configs/
COPY tests/ tests/
RUN python -m pip install --no-deps -e . \
    && python -m pip check

CMD ["python", "-m", "pytest", "-q"]

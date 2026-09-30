FROM python:3.12-slim
WORKDIR /app

ARG FINDX_INSTALL_FULL=0
COPY backend/requirements.txt /tmp/requirements.txt
COPY backend/requirements-full.txt /tmp/requirements-full.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && if [ "$FINDX_INSTALL_FULL" = "1" ]; then pip install --no-cache-dir -r /tmp/requirements-full.txt; fi

COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

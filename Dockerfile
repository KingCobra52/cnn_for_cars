# Serving image. Deliberately does not install torch: the exported ONNX graph carries
# the whole model, so the runtime needs only onnxruntime and PIL. That is the difference
# between a ~150 MB image and a ~2 GB one.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Only app/ is needed: serving.py is self-contained and imports no torch and no
# carvision. tests/test_serving_parity.py pins its preprocessing to the training one.
COPY app/ ./

# The serving bundle (model.onnx, classes.txt, serving.json) is mounted at runtime
# rather than baked in, so the image does not have to be rebuilt to ship a new model:
#   docker run -v $(pwd)/artifacts/serving:/srv/model -p 8000:8000 carvision
ENV CARVISION_MODEL_DIR=/srv/model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

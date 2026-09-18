"""A minimal FastAPI service around the exported graph.

The Gradio demo is for people; this is for programs. It exists to show the model can be
put behind a typed HTTP contract, not because the project needs two front ends.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Any

from carvision.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(model_dir: Path | None = None) -> Any:
    """Build the FastAPI application.

    Args:
        model_dir: Directory holding the serving bundle. Defaults to the demo's location.

    Returns:
        The configured FastAPI app.
    """
    from fastapi import FastAPI, HTTPException
    from PIL import Image, UnidentifiedImageError
    from pydantic import BaseModel, Field

    import app.app as demo  # noqa: PLC0415 -- deferred so importing this module is cheap

    configure_logging()
    predictor = demo.Predictor(model_dir or demo.MODEL_DIR)

    class Prediction(BaseModel):
        """One candidate class."""

        label: str = Field(description="Class name, e.g. '2012 Tesla Model S Sedan'.")
        probability: float = Field(ge=0.0, le=1.0, description="Calibrated probability.")

    class PredictResponse(BaseModel):
        """The service's reply."""

        predictions: list[Prediction] = Field(description="Top candidates, best first.")
        backbone: str = Field(description="Frozen backbone that produced the features.")

    class PredictRequest(BaseModel):
        """A base64-encoded image."""

        image_base64: str = Field(description="Base64-encoded JPEG or PNG bytes.")

    api = FastAPI(
        title="carvision",
        version=predictor.config.get("run", "unknown"),
        description="Fine-grained car classification over the 196 Stanford Cars classes.",
    )

    @api.get("/health")
    def health() -> dict[str, str | int]:
        """Liveness probe, also reporting which model is loaded."""
        return {
            "status": "ok",
            "backbone": predictor.config.get("backbone", "unknown"),
            "num_classes": len(predictor.class_names),
        }

    @api.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Classify one image.

        Args:
            request: The base64-encoded image.

        Returns:
            The top predictions with calibrated probabilities.

        Raises:
            HTTPException: 400 if the payload is not decodable as an image.
        """
        try:
            raw = base64.b64decode(request.image_base64, validate=True)
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as error:
            raise HTTPException(
                status_code=400, detail=f"Could not decode image: {error}"
            ) from error

        scores = predictor(image)
        return PredictResponse(
            predictions=[
                Prediction(label=label, probability=probability)
                for label, probability in scores.items()
            ],
            backbone=str(predictor.config.get("backbone", "unknown")),
        )

    return api

"""API Routes"""
from .predict import router as predict_router
from .models import router as models_router
from .cache import router as cache_router
from .health import router as health_router

__all__ = ["predict_router", "models_router", "cache_router", "health_router"]

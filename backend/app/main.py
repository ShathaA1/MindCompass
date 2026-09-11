"""Creates and configures the main FastAPI application."""

from fastapi import FastAPI
from app.api import auth, learners, learning

app = FastAPI(
    title="MindCompass API",
    description="Backend API for the MindCompass AI tutoring agent.",
    version="0.1.0",
)

app.include_router(auth.router)
app.include_router(learners.router)
app.include_router(learning.router)

@app.get("/")
def root():
    """Return a basic API welcome message."""
    return {"message": "MindCompass API is running"}

@app.get("/health")
def health_check():
    """Return the current health status of the API."""
    return {"status": "healthy"}
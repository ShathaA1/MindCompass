"""Creates and configures the main FastAPI application."""

from fastapi import FastAPI

app = FastAPI(
    title="MindCompass API",
    description="Backend API for the MindCompass AI tutoring agent.",
    version="0.1.0",
)


@app.get("/")
def root():
    """Return a basic API welcome message."""

    return {"message": "MindCompass API is running"}


@app.get("/health")
def health_check():
    """Return the current health status of the API."""

    return {"status": "healthy"}

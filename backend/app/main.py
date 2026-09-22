"""Creates and configures the main FastAPI application."""

from fastapi import FastAPI

from app.api import auth, learners, learning, chat


app = FastAPI(
    title="MindCompass API",
    description="Backend API for the MindCompass AI tutoring agent.",
    version="0.1.0",
)


# Register authentication endpoints.
app.include_router(auth.router)

# Register learner profile endpoints.
app.include_router(learners.router)

# Register learning path and diagnostic endpoints.
app.include_router(learning.router)

# Register Tutor Agent chat endpoints.
app.include_router(chat.router)


@app.get("/")
def root():
    """Return a basic API welcome message."""

    return {
        "message": "MindCompass API is running"
    }


@app.get("/health")
def health_check():
    """Return the current health status of the API."""

    return {
        "status": "healthy"
    }
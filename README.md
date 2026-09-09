# MindCompass

MindCompass is an agentic AI tutoring system that creates and
continuously adapts personalized learning paths based on each
learner's goals, current level, preferences, and performance.

## Learning Cycle

Understand → Plan → Teach → Assess → Adapt → Remember

## Technology Stack

- Python
- FastAPI
- Streamlit
- PostgreSQL
- ChromaDB
- OpenAI
- LangChain
- LangGraph

## Project Structure

- `backend/app/database/` — PostgreSQL connection and models
- `backend/app/api/` — FastAPI endpoints
- `backend/app/services/` — Application business logic
- `backend/app/rag/` — Knowledge-base ingestion and retrieval
- `backend/app/agent/` — Tutor Agent workflow and tools
- `frontend/` — Streamlit learner interface
- `knowledge_base/` — Approved educational source materials
- `scripts/` — Topic seeding and material ingestion
- `tests/` — API, RAG, and Agent tests

## Status

Development in progress.

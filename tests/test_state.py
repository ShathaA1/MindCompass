from app.agent.state import TutorState


def test_tutor_state():
    state: TutorState = {
        "user_id": 1,
        "session_id": 1,
        "user_message": "What should I learn next?"
    }

    assert state["user_id"] == 1
    assert state["session_id"] == 1
    assert state["user_message"] == "What should I learn next?"
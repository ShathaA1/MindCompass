from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database.connection import get_db
from app.database.models import User
from app.schemas.learner import LoginRequest, RegisterRequest


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db),
):
    existing_user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    new_user = User(
        name=data.name,
        email=data.email,
        password_hash=hash_password(data.password),
        role="learner",
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "user_id": new_user.user_id,
        "name": new_user.name,
        "email": new_user.email,
    }


@router.post("/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if not user or not verify_password(
        data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    token = create_access_token(
        {
            "sub": str(user.user_id),
            "email": user.email,
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer",
    }


@router.get("/me")
def me(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(
            User.user_id == int(current_user["sub"])
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    return {
        "user_id": user.user_id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }

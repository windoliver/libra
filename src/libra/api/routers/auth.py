"""
Authentication Router for LIBRA API (Issue #30).

Endpoints:
- POST /token - Get JWT token
- GET /me - Get current user info
- POST /api-key - Create API key
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from libra.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    create_api_key,
)
from libra.api.deps import get_current_active_user
from libra.api.schemas import Token, UserResponse


router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """
    OAuth2 compatible token login.

    Get an access token using username and password.

    **Demo credentials:**
    - admin/admin123 (full access)
    - trader/trader123 (read/write)
    - viewer/viewer123 (read only)
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["username"], "scopes": user.get("scopes", [])},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[dict, Depends(get_current_active_user)],
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user.get("email"),
        is_active=current_user.get("is_active", True),
        created_at=current_user.get("created_at", datetime.now(timezone.utc)),
    )


@router.post("/api-key")
async def create_new_api_key(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    name: str = "default",
) -> dict:
    """
    Create a new API key for the current user.

    API keys can be used as an alternative to JWT tokens.
    Pass the key in the X-API-Key header.
    """
    api_key = create_api_key(current_user["id"], name)
    return {
        "api_key": api_key,
        "name": name,
        "message": "Store this key securely. It won't be shown again.",
    }

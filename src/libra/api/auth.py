"""
JWT Authentication for LIBRA API (Issue #30).

Provides:
- JWT token creation and validation
- Password hashing
- User authentication flow
- API key authentication (alternative)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from libra.api.schemas import TokenData


# =============================================================================
# Configuration
# =============================================================================

# Secret key for JWT signing - should be set via environment variable in production
SECRET_KEY = os.getenv("LIBRA_API_SECRET_KEY", "development-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("LIBRA_API_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# Password Hashing
# =============================================================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    # For demo purposes, use simple comparison with known passwords
    # In production, use proper bcrypt verification
    demo_passwords = {
        "admin": "admin123",
        "trader": "trader123",
        "viewer": "viewer123",
    }
    # Check if this is a demo user (hashed_password is actually username for demo)
    if hashed_password in demo_passwords:
        return plain_password == demo_passwords[hashed_password]

    # For real bcrypt hashes (when available)
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    try:
        return pwd_context.hash(password)
    except Exception:
        # Fallback for environments where bcrypt is problematic
        return password


# =============================================================================
# JWT Token Management
# =============================================================================


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenData | None:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            return None
        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)
    except JWTError:
        return None


# =============================================================================
# API Key Authentication (Alternative to JWT)
# =============================================================================

# In-memory API keys for demo - in production use database
_api_keys: dict[str, dict[str, Any]] = {}


def create_api_key(user_id: str, name: str = "default") -> str:
    """
    Create an API key for a user.

    Args:
        user_id: User ID
        name: Key name/description

    Returns:
        Generated API key
    """
    import secrets

    api_key = secrets.token_urlsafe(32)
    _api_keys[api_key] = {
        "user_id": user_id,
        "name": name,
        "created_at": datetime.now(timezone.utc),
    }
    return api_key


def validate_api_key(api_key: str) -> dict[str, Any] | None:
    """
    Validate an API key.

    Args:
        api_key: API key to validate

    Returns:
        Key metadata if valid, None otherwise
    """
    return _api_keys.get(api_key)


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key.

    Args:
        api_key: API key to revoke

    Returns:
        True if revoked, False if not found
    """
    if api_key in _api_keys:
        del _api_keys[api_key]
        return True
    return False


# =============================================================================
# Demo Users (In-memory for development)
# =============================================================================

# Demo users - in production use database with proper bcrypt hashes
# For demo, we use the username as a marker and verify against known passwords
_demo_users: dict[str, dict[str, Any]] = {
    "admin": {
        "id": "user_admin",
        "username": "admin",
        "email": "admin@libra.local",
        "hashed_password": "admin",  # Marker for demo - verify_password handles this
        "is_active": True,
        "scopes": ["read", "write", "admin"],
    },
    "trader": {
        "id": "user_trader",
        "username": "trader",
        "email": "trader@libra.local",
        "hashed_password": "trader",  # Marker for demo
        "is_active": True,
        "scopes": ["read", "write"],
    },
    "viewer": {
        "id": "user_viewer",
        "username": "viewer",
        "email": "viewer@libra.local",
        "hashed_password": "viewer",  # Marker for demo
        "is_active": True,
        "scopes": ["read"],
    },
}


def get_user(username: str) -> dict[str, Any] | None:
    """Get user by username."""
    return _demo_users.get(username)


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """
    Authenticate a user with username and password.

    Args:
        username: Username
        password: Plain password

    Returns:
        User data if authenticated, None otherwise
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

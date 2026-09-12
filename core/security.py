import os
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

# Configuration
INSECURE_DEVELOPMENT_SECRET = "CHANGE-THIS-TO-A-RANDOM-SECRET-KEY"
SECRET_KEY = os.getenv("JWT_SECRET", INSECURE_DEVELOPMENT_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "1"))
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(ACCESS_TOKEN_EXPIRE_HOURS * 60))
)
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
ACCESS_COOKIE_NAME = os.getenv("ACCESS_COOKIE_NAME", "admin_access")
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "admin_refresh")
GRAPHICS_ACCESS_COOKIE_NAME = os.getenv("GRAPHICS_ACCESS_COOKIE_NAME", "graphics_access")
GRAPHICS_REFRESH_COOKIE_NAME = os.getenv("GRAPHICS_REFRESH_COOKIE_NAME", "graphics_refresh")
AUTH_COOKIE_SECURE = os.getenv(
    "AUTH_COOKIE_SECURE",
    "true" if os.getenv("ENVIRONMENT", "development").lower() == "production" else "false",
).lower() in {"1", "true", "yes"}
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
AUTH_COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN") or None

# The request middleware sets this for existing routes that still call
# verify_token(token). Explicit query/body tokens continue to take precedence.
_request_token: ContextVar[Optional[str]] = ContextVar("admin_request_token", default=None)
_graphics_request_token: ContextVar[Optional[str]] = ContextVar("graphics_request_token", default=None)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(plain_password: str) -> str:
    """Hash a new password (argon2, same context used to verify)"""
    return pwd_context.hash(plain_password)

def password_strength_error(password: str) -> Optional[str]:
    """Return a human-readable reason a password is too weak, or None if it passes."""
    if len(password) < 12:
        return "Password must be at least 12 characters long"
    if not any(char.isalpha() for char in password):
        return "Password must include at least one letter"
    if not any(char.isdigit() for char in password):
        return "Password must include at least one number"
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a refresh JWT with a distinct token type and longer lifetime."""
    payload = data.copy()
    payload["type"] = "refresh"
    return create_access_token(
        payload,
        expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

def set_request_token(token: Optional[str]):
    return _request_token.set(token)

def reset_request_token(token_context):
    _request_token.reset(token_context)

def verify_token(token: Optional[str] = None, expected_type: str = "access") -> Optional[str]:
    """Verify a JWT token and return the email"""
    token = token or _request_token.get()
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type", "access") != expected_type:
            return None
        email: str = payload.get("sub")
        return email
    except JWTError:
        return None

def set_graphics_request_token(token: Optional[str]):
    return _graphics_request_token.set(token)

def reset_graphics_request_token(token_context):
    _graphics_request_token.reset(token_context)

def create_graphics_access_token(email: str, department_id: int, user_id: int) -> str:
    return create_access_token(
        {"sub": email, "department_id": department_id, "user_id": user_id, "type": "graphics_access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

def create_graphics_refresh_token(email: str, department_id: int, user_id: int) -> str:
    return create_access_token(
        {"sub": email, "department_id": department_id, "user_id": user_id, "type": "graphics_refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

def verify_graphics_token(token: Optional[str] = None, expected_type: str = "graphics_access") -> Optional[dict]:
    """Verify a graphics session JWT and return its payload (email/department_id/user_id)."""
    token = token or _graphics_request_token.get()
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != expected_type:
            return None
        return payload
    except JWTError:
        return None

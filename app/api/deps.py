import hashlib

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import User
from app.db.session import get_db


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_current_user(x_api_key: str = Header(..., alias="X-API-Key"),
                     db: Session = Depends(get_db)) -> User:
    user = repo.get_user_by_key_hash(db, hash_key(x_api_key))
    if user is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return user

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bookaudio.models import Base


def die(message: str):
    raise RuntimeError(message)

def database_url():
    user = os.getenv("DATABASE_USER") or "bookaudio_app"
    password = os.getenv("DATABASE_PASSWORD") or die("DATABASE_PASSWORD environment variable not set")
    host = os.getenv("DATABASE_HOST") or "localhost"
    port = os.getenv("DATABASE_PORT") or 5432
    return (
        f"postgresql+asyncpg://{user}:{password}@{host}:{port}/bookaudio"
    )

engine = create_async_engine(
    database_url(),
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create tables if they don't already exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

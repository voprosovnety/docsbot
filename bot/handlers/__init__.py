from aiogram import Router

from . import admin, documents, questions, start


def build_router() -> Router:
    """Order matters: commands and documents are matched before free text."""
    router = Router(name="root")
    router.include_router(start.router)
    router.include_router(admin.router)
    router.include_router(documents.router)
    router.include_router(questions.router)
    return router


__all__ = ["build_router"]

"""Shared call-to-action card configuration for the website and mobile app."""

import sqlite3
from contextlib import closing
from typing import Annotated

from pydantic import AnyHttpUrl, BaseModel, StringConstraints, field_validator

from core.database import get_db_path


class HomepageCardContent(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    buttonLabel: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]
    url: AnyHttpUrl

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: AnyHttpUrl) -> AnyHttpUrl:
        if url.scheme != "https" or url.username is not None or url.password is not None:
            raise ValueError("Use an HTTPS link without credentials.")
        return url


class HomepageCardSettings(HomepageCardContent):
    enabled: bool


def get_homepage_card_settings() -> HomepageCardSettings:
    with closing(sqlite3.connect(get_db_path())) as connection:
        row = connection.execute(
            "SELECT enabled, title, message, button_label, url FROM homepage_card WHERE id = 1"
        ).fetchone()
    if row is None:
        raise sqlite3.DatabaseError("Homepage card settings have not been initialized")
    enabled, title, message, button_label, url = row
    return HomepageCardSettings(enabled=bool(enabled), title=title, message=message, buttonLabel=button_label, url=url)


def save_homepage_card(settings: HomepageCardSettings) -> None:
    with closing(sqlite3.connect(get_db_path())) as connection, connection:
        connection.execute(
            "UPDATE homepage_card SET enabled = ?, title = ?, message = ?, button_label = ?, url = ? WHERE id = 1",
            (settings.enabled, settings.title, settings.message, settings.buttonLabel, str(settings.url)),
        )


def get_public_homepage_card() -> dict | None:
    settings = get_homepage_card_settings()
    return settings.model_dump(mode="json", exclude={"enabled"}) if settings.enabled else None

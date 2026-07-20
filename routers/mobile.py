import os
import re
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field, field_validator

from services.mobile_content import build_mobile_content, county_catalog
from services.mobile_push import disable_subscription, upsert_subscription

router = APIRouter(prefix="/api/mobile", tags=["mobile"])
EXPO_TOKEN_PATTERN = re.compile(r"^(?:Expo|Exponent)PushToken\[[A-Za-z0-9_-]+\]$")


class NotificationPreferences(BaseModel):
    forecast: bool = False
    sitrep: bool = False
    fireWeather: bool = False
    countyFips: list[str] = Field(default_factory=list, max_length=115)

    @field_validator("countyFips")
    @classmethod
    def validate_counties(cls, values: list[str]) -> list[str]:
        known = {county["fips"] for county in county_catalog()}
        normalized = sorted(set(values))
        unknown = [value for value in normalized if value not in known]
        if unknown:
            raise ValueError(f"Unknown Missouri county FIPS: {', '.join(unknown[:3])}")
        return normalized


class PushSubscriptionRequest(BaseModel):
    expoPushToken: str = Field(min_length=20, max_length=256)
    platform: Literal["ios", "android"]
    appVersion: str = Field(default="", max_length=64)
    preferences: NotificationPreferences

    @field_validator("expoPushToken")
    @classmethod
    def validate_push_token(cls, value: str) -> str:
        if not EXPO_TOKEN_PATTERN.fullmatch(value):
            raise ValueError("Invalid Expo push token")
        return value


@router.get("/content")
def get_content():
    api_base_url = os.getenv("PUBLIC_API_BASE_URL", "https://api.showmefire.org")
    cdn_base_url = os.getenv("PUBLIC_CDN_BASE_URL", "https://cdn.showmefire.org/latest")
    return build_mobile_content(api_base_url, cdn_base_url)


@router.put("/push-subscriptions/{installation_id}")
def put_push_subscription(installation_id: UUID, payload: PushSubscriptionRequest):
    return upsert_subscription(
        installation_id=str(installation_id),
        expo_push_token=payload.expoPushToken,
        platform=payload.platform,
        app_version=payload.appVersion,
        forecast=payload.preferences.forecast,
        sitrep=payload.preferences.sitrep,
        fire_weather=payload.preferences.fireWeather,
        county_fips=payload.preferences.countyFips,
    )


@router.delete("/push-subscriptions/{installation_id}", status_code=204)
def delete_push_subscription(installation_id: UUID):
    disable_subscription(str(installation_id))
    return Response(status_code=204)

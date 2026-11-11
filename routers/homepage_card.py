from fastapi import APIRouter, Depends, Response

from core.admin_auth.permissions import HOMEPAGE_CARD_MANAGE, require_permission
from services.homepage_card import (
    HomepageCardSettings,
    get_homepage_card_settings,
    get_public_homepage_card,
    save_homepage_card,
)

router = APIRouter(tags=["homepage-card"])


@router.get("/api/homepage-card")
def public_homepage_card(response: Response):
    response.headers["Cache-Control"] = "public, max-age=60"
    return {"card": get_public_homepage_card()}


@router.get("/api/admin/homepage-card", dependencies=[Depends(require_permission(HOMEPAGE_CARD_MANAGE))])
def admin_get_homepage_card(response: Response) -> HomepageCardSettings:
    response.headers["Cache-Control"] = "no-store"
    return get_homepage_card_settings()


@router.put("/api/admin/homepage-card", dependencies=[Depends(require_permission(HOMEPAGE_CARD_MANAGE))])
def admin_save_homepage_card(settings: HomepageCardSettings) -> HomepageCardSettings:
    save_homepage_card(settings)
    return settings

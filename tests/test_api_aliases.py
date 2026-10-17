from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from core.api_aliases import add_unprefixed_aliases


def _deny():
    raise HTTPException(status_code=401, detail="nope")


def _app(tmp_path):
    app = FastAPI()

    @app.get("/api/fires")
    def fires():
        return {"v": "fires"}

    @app.get("/api/fires/{fire_id}")
    def fire(fire_id: str):
        return {"id": fire_id}

    @app.post("/api/posts")
    def post_create():
        return {"v": "created"}

    @app.get("/api/secure", dependencies=[Depends(_deny)])
    def secure():
        return {"v": "secret"}

    @app.get("/api/admin/thing")
    def admin():
        return {"v": "admin"}

    @app.get("/api/graphics/auth/logout")
    def logout():
        return {"v": "logout"}

    @app.get("/api/conditions")
    def conditions():
        return {"v": "api-conditions"}

    @app.get("/conditions")
    def conditions_existing():
        return {"v": "existing"}

    @app.get("/stations/mine")
    def literal_existing():
        return {"v": "existing-literal"}

    @app.get("/api/stations/{sid}")
    def station(sid: str):
        return {"v": "api-station"}

    @app.get("/api/images")
    def images_listing():
        return {"v": "listing"}

    (tmp_path / "x.txt").write_text("hi")
    app.mount("/images", StaticFiles(directory=str(tmp_path)), name="images")
    app.mount("/", StaticFiles(directory=str(tmp_path)), name="public")
    report = add_unprefixed_aliases(app)  # root mount already present: aliases must still win
    return TestClient(app), report


def test_aliases_work_and_originals_stay(tmp_path):
    client, report = _app(tmp_path)
    assert client.get("/fires").json() == {"v": "fires"}
    assert client.get("/api/fires").json() == {"v": "fires"}
    assert client.get("/fires/7").json() == {"id": "7"}
    assert client.post("/posts").json() == {"v": "created"}


def test_alias_keeps_dependencies(tmp_path):
    client, _ = _app(tmp_path)
    assert client.get("/api/secure").status_code == 401
    assert client.get("/secure").status_code == 401


def test_admin_and_graphics_auth_not_aliased(tmp_path):
    client, report = _app(tmp_path)
    assert client.get("/admin/thing").status_code == 404
    assert client.get("/graphics/auth/logout").status_code == 404
    assert client.get("/api/admin/thing").status_code == 200


def test_collisions_are_skipped_and_reported(tmp_path):
    client, report = _app(tmp_path)
    assert client.get("/conditions").json() == {"v": "existing"}
    assert client.get("/api/conditions").json() == {"v": "api-conditions"}
    assert client.get("/stations/mine").json() == {"v": "existing-literal"}
    skipped = {i["path"]: i["reason"] for i in report["skipped"]}
    assert "/api/conditions" in skipped
    # A param alias added after an existing literal route cannot shadow it.
    assert client.get("/stations/abc").json() == {"v": "api-station"}
    assert skipped["/api/images"].startswith("mount")


def test_aliases_hidden_from_schema(tmp_path):
    client, _ = _app(tmp_path)
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/fires" in paths and "/fires" not in paths

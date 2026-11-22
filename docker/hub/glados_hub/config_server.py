"""
Live configuration for glados_hub — web UI + validated REST endpoints.

The device inventory (Kasa lights, camera policy) and her system prompt
are plain files. They used to be baked into the Docker image, so any
change meant a full `docker compose build`. docker-compose.yml now
bind-mounts the config + prompts directories, and this module makes the
hub its own editor:

  GET  /                      the configuration UI (self-contained HTML)
  GET  /api/config/lights     current kasa.json
  PUT  /api/config/lights     validate + write + swap the live registry
  GET  /api/config/cameras    current cameras.json
  PUT  /api/config/cameras    validate + write + re-apply the monitor
  GET  /api/config/prompt     current glados_system.txt (base personality)
  PUT  /api/config/prompt     write + rebuild the system prompt
  GET  /api/config/volume     per-device speaker volume
  PUT  /api/config/volume     set one device's volume (pushed live)
  GET  /api/config/status     camera feed + light inventory summary

Every PUT is validated BEFORE anything is written, and the write itself
is atomic (temp file + fsync + rename), so a crash or a bad payload can
never corrupt a config. Keys starting with "_" (e.g. the cameras.json
_doc) are preserved across edits.

This module stays pure on purpose: it imports only the leaf capability
modules (kasa_lights, cameras) — never the hub — so it unit-tests in a
bare environment. The hub wires in the two things it owns (system-prompt
rebuild, volume push) via set_on_changed() / set_volume_api().

Security: like /health and /say, the config surface is LAN-trust OPEN by
default. Set GLADOS_CONFIG_TOKEN (docker-compose.yml) to require the
token as ?token=... or the X-Config-Token header on every endpoint,
including the UI page itself.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from . import cameras
from . import kasa_lights
from . import quiet

logger = logging.getLogger("glados_hub.config")

# Editable files. kasa/cameras paths honor their env overrides (and, in
# the container, point at the bind-mounted host directory); the prompt
# path is the baked prompts/ dir (also bind-mounted in production).
KASA_PATH = kasa_lights.CONFIG_PATH
CAMERAS_PATH = cameras.CONFIG_PATH
QUIET_PATH = quiet.CONFIG_PATH
PROMPT_PATH = Path(__file__).parent / "prompts" / "glados_system.txt"
UI_PATH = Path(__file__).parent / "config_ui.html"

TOKEN = os.environ.get("GLADOS_CONFIG_TOKEN", "")
PROMPT_MAX_CHARS = 60000


class ConfigError(ValueError):
    """User-submitted config failed validation."""


# ── file I/O ─────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path.name} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: top level must be a JSON object")
    return raw


def _atomic_write_json(path: Path, obj: dict) -> None:
    """temp file + fsync + rename in the same dir: a crash mid-write can
    never leave a truncated/corrupt config behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _preserve_doc(existing: dict, incoming: dict) -> dict:
    """Keep '_'-prefixed keys (documentation) across an edit: the file's
    current ones first, any the client sent win on conflict."""
    out: dict = {k: v for k, v in existing.items() if k.startswith("_")}
    for k, v in incoming.items():
        if k.startswith("_"):
            out[k] = v
    return out


# ── lights (kasa.json) ───────────────────────────────────────────────


def read_lights() -> dict:
    return _read_json(KASA_PATH)


def _validate_lights(raw: dict) -> None:
    lights, types, groups = raw.get("lights", {}), raw.get("types", {}), raw.get("groups", {})
    for field, val in (("lights", lights), ("types", types), ("groups", groups)):
        if not isinstance(val, dict):
            raise ConfigError(f"'{field}' must be an object")
    for name, ip in lights.items():
        if not isinstance(ip, str) or not ip.strip():
            raise ConfigError(f"light '{name}': ip must be a non-empty string")
    for name, t in types.items():
        if t not in ("plug", "bulb", "light"):
            raise ConfigError(f"type for '{name}' must be one of plug | bulb | light")
    for g, members in groups.items():
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            raise ConfigError(f"group '{g}': members must be a list of light names")
        for m in members:
            if m not in lights:
                raise ConfigError(f"group '{g}': member '{m}' is not a configured light")


def write_lights(raw: dict) -> dict:
    """Validate + atomically write kasa.json; returns the written config."""
    if not isinstance(raw, dict):
        raise ConfigError("top level must be a JSON object")
    _validate_lights(raw)
    out = _preserve_doc(_read_json(KASA_PATH), raw)
    for key in ("lights", "types", "groups"):
        if key in raw:
            out[key] = raw[key]
    _atomic_write_json(KASA_PATH, out)
    return out


# ── cameras (cameras.json) ───────────────────────────────────────────


def read_cameras() -> dict:
    return _read_json(CAMERAS_PATH)


def _validate_cameras(raw: dict) -> None:
    cams = raw.get("cameras", {})
    if not isinstance(cams, dict):
        raise ConfigError("'cameras' must be an object")
    for name, spec in cams.items():
        where = f"camera '{name}'"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: must be an object")
        if not spec.get("id"):
            raise ConfigError(f"{where}: 'id' (the Scrypted camera id) is required")
        for field in ("classes", "zones"):
            val = spec.get(field, [])
            if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                raise ConfigError(f"{where}: '{field}' must be a list of strings")
        cd = spec.get("cooldown_s", 90)
        if isinstance(cd, bool) or not isinstance(cd, (int, float)) or cd < 0:
            raise ConfigError(f"{where}: 'cooldown_s' must be a number >= 0")
        if "announce" in spec and not isinstance(spec["announce"], bool):
            raise ConfigError(f"{where}: 'announce' must be true or false")
        lines = spec.get("lines", {})
        if not isinstance(lines, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in lines.items()
        ):
            raise ConfigError(f"{where}: 'lines' must be an object of strings")


def write_cameras(raw: dict) -> dict:
    """Validate + atomically write cameras.json; returns the written config."""
    if not isinstance(raw, dict):
        raise ConfigError("top level must be a JSON object")
    _validate_cameras(raw)
    out = _preserve_doc(_read_json(CAMERAS_PATH), raw)
    if "cameras" in raw:
        out["cameras"] = raw["cameras"]
    _atomic_write_json(CAMERAS_PATH, out)
    return out


# ── system prompt (glados_system.txt) ────────────────────────────────


def read_prompt() -> str:
    return PROMPT_PATH.read_text()


def write_prompt(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ConfigError("system prompt must be a non-empty string")
    if len(text) > PROMPT_MAX_CHARS:
        raise ConfigError(f"system prompt too long ({len(text)} > {PROMPT_MAX_CHARS} chars)")
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(text)
    return text


# ── apply (in-memory hot reload) ─────────────────────────────────────


def apply_lights() -> str:
    """Swap the live Kasa registry from the file just written."""
    reg = kasa_lights.reload()
    return reg.describe()


def apply_cameras() -> bool:
    """Re-read cameras.json and re-apply the live monitor (re-subscribe)."""
    return cameras.reload()


# ── quiet hours (quiet_hours.json) ──────────────────────────────────


def read_quiet() -> dict:
    return _read_json(QUIET_PATH)


def write_quiet(raw: dict) -> dict:
    """Validate + atomically write quiet_hours.json; returns the written
    config. The client may send just {"timezone", "windows"} — _doc is
    preserved from the file."""
    if not isinstance(raw, dict):
        raise ConfigError("top level must be a JSON object")
    try:
        quiet.validate(raw)
    except quiet.QuietConfigError as e:
        raise ConfigError(str(e)) from e
    out = _preserve_doc(_read_json(QUIET_PATH), raw)
    for key in ("timezone", "windows"):
        if key in raw:
            out[key] = raw[key]
    _atomic_write_json(QUIET_PATH, out)
    return out


def apply_quiet() -> dict:
    """Re-read quiet_hours.json; the enforcement points check it live."""
    quiet.reload()
    return quiet.status()


# ── hub-injected seams (avoid importing the hub) ─────────────────────

# () -> None — called after any successful config write; the hub rebuilds
# SYSTEM_PROMPT (personality file + live capability block) and pushes it
# to in-flight sessions.
_on_changed = None

# The hub's volume store + pusher: .list() -> {device: level},
# async .set(device, level) -> level.
_volume_api = None


def set_on_changed(fn) -> None:
    global _on_changed
    _on_changed = fn


def set_volume_api(api) -> None:
    global _volume_api
    _volume_api = api




# ── REST ─────────────────────────────────────────────────────────────


async def _require_token(
    token: str = Query(default=""),
    x_config_token: str = Header(default=""),
) -> None:
    if not TOKEN:
        return  # auth disabled: LAN-trust, same as /health and /say
    if token == TOKEN or x_config_token == TOKEN:
        return
    raise HTTPException(status_code=401, detail="invalid or missing config token")


router = APIRouter(dependencies=[Depends(_require_token)], tags=["config"])


@router.get("/", response_class=HTMLResponse)
async def ui() -> str:
    return UI_PATH.read_text()


@router.get("/api/config/lights")
async def get_lights() -> dict:
    return read_lights()


@router.put("/api/config/lights")
async def put_lights(payload: dict) -> dict:
    try:
        written = write_lights(payload)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    inventory = apply_lights()
    logger.info("config: lights saved from UI — %s", inventory)
    if _on_changed:
        _on_changed()
    return {"ok": True, "config": written}


@router.get("/api/config/cameras")
async def get_cameras() -> dict:
    return read_cameras()


@router.put("/api/config/cameras")
async def put_cameras(payload: dict) -> dict:
    try:
        written = write_cameras(payload)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    running = apply_cameras()  # re-subscribes; the feed blips briefly
    logger.info("config: cameras saved from UI (monitor running=%s)", running)
    if _on_changed:
        _on_changed()
    return {"ok": True, "config": written, "monitor_running": running}


@router.get("/api/config/quiet")
async def get_quiet() -> dict:
    out = read_quiet()
    out["status"] = quiet.status()
    return out


@router.put("/api/config/quiet")
async def put_quiet(payload: dict) -> dict:
    try:
        written = write_quiet(payload)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    st = apply_quiet()
    logger.info("config: quiet hours saved from UI — %s",
                ", ".join(w["name"] for w in written.get("windows", []))
                or "no windows")
    if _on_changed:
        _on_changed()
    return {"ok": True, "config": written, "status": st}


@router.get("/api/config/prompt")
async def get_prompt() -> dict:
    return {"prompt": read_prompt()}


@router.put("/api/config/prompt")
async def put_prompt(payload: dict) -> dict:
    try:
        write_prompt(str(payload.get("prompt", "")))
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    logger.info("config: system prompt saved from UI")
    if _on_changed:
        _on_changed()
    return {"ok": True}


@router.get("/api/config/volume")
async def get_volume() -> dict:
    return _volume_api.list() if _volume_api else {}


@router.put("/api/config/volume")
async def put_volume(payload: dict) -> dict:
    if _volume_api is None:
        raise HTTPException(status_code=503, detail="volume control not wired (hub not started)")
    device = str(payload.get("device_id", "")).strip()
    if not device:
        raise HTTPException(status_code=400, detail="device_id is required")
    try:
        level = int(payload.get("level"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="level must be an integer 0-100") from None
    new = await _volume_api.set(device, level)
    return {"ok": True, "device_id": device, "level": new}


@router.get("/api/config/status")
async def status() -> dict:
    return {
        "cameras": cameras.status(),
        "cameras_configured": sorted(cameras.CONFIG.cameras),
        "lights": kasa_lights.REGISTRY.describe(),
        "quiet": quiet.status(),
    }
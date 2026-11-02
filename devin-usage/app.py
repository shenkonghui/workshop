#!/usr/bin/env python3
"""Devin Usage — a lightweight Flask dashboard for Devin CLI session token usage.

Reads directly from the local sessions.db (read-only) and serves:
  GET /            — dashboard page
  GET /api/sessions?days=1   — session list + aggregates
  GET /api/by-model?days=7   — token usage grouped by model
  GET /api/by-day?days=30    — token usage grouped by day
"""
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta

import yaml
from flask import Flask, jsonify, render_template, request

DB_PATH = os.path.expanduser("~/.local/share/devin/cli/sessions.db")

app = Flask(__name__)

# Per-token pricing (USD per 1M tokens), keyed by the `model` column in
# sessions.db.  Loaded from prices.yaml (single source of truth); the inline
# dict below is only a fallback if the yaml cannot be read.
# Sources: OpenAI / Anthropic / Z.ai / DeepSeek / Moonshot / Cognition official pages.
_PRICING_FALLBACK = {
    "glm-5-2":                  {"input": 1.40,  "cached": 0.26,   "cache_write": 0.0,   "output": 4.40},
    "glm-5-3-flash-high":       {"input": 0.15,  "cached": 0.01,   "cache_write": 0.0,   "output": 0.50},
    "gpt-5-6-sol-high":         {"input": 1.50,  "cached": 0.15,   "cache_write": 1.875, "output": 9.00},
    "gpt-5-6-luna-medium":      {"input": 0.20,  "cached": 0.02,   "cache_write": 0.25,  "output": 1.20},
    "deepseek-v4-pro-high":     {"input": 1.74,  "cached": 0.15,   "cache_write": 0.0,   "output": 3.48},
    "deepseek-v4-flash-high":   {"input": 0.14,  "cached": 0.028,  "cache_write": 0.0,   "output": 0.28},
    "kimi-k3-high":             {"input": 3.00,  "cached": 0.30,   "cache_write": 0.0,   "output": 15.00},
    "claude-sonnet-4-6":        {"input": 3.00,  "cached": 0.30,   "cache_write": 3.75,  "output": 15.00},
    "swe-1-7-medium":           {"input": 0.50,  "cached": 0.20,   "cache_write": 0.0,   "output": 2.50},
    "swe-1-6-fast":             {"input": 0.50,  "cached": 0.20,   "cache_write": 0.0,   "output": 2.50},
    "swe-1-7-lightning-medium": {"input": 2.50,  "cached": 1.00,   "cache_write": 0.0,   "output": 12.50},
    "swe-2-high":               {"input": 0.75,  "cached": 0.075,  "cache_write": 0.0,   "output": 3.75},
    "swe-2-max":                {"input": 0.75,  "cached": 0.075,  "cache_write": 0.0,   "output": 3.75},
    "swe-2-medium":             {"input": 0.75,  "cached": 0.075,  "cache_write": 0.0,   "output": 3.75},
    "glm-5-3-high":             {"input": 1.40,  "cached": 0.26,   "cache_write": 0.0,   "output": 4.40},
    "glm-5-3-flash-max":        {"input": 0.15,  "cached": 0.029,  "cache_write": 0.0,   "output": 0.50},
    "gemini-3-8-flash-high":    {"input": 1.50,  "cached": 0.15,   "cache_write": 1.50,  "output": 7.50},
    "gemini-3-8-flash-medium":  {"input": 1.50,  "cached": 0.15,   "cache_write": 1.50,  "output": 7.50},
    "claude-opus-5-medium":     {"input": 5.00,  "cached": 0.50,   "cache_write": 6.25,  "output": 25.00},
    "gpt-5-6-sol-medium":       {"input": 4.00,  "cached": 0.40,   "cache_write": 5.00,  "output": 20.00},
    "deepseek-v4-1-flash-high": {"input": 0.14,  "cached": 0.028,  "cache_write": 0.0,   "output": 0.28},
}


def _load_pricing() -> dict:
    """Load MODEL_PRICING from prices.yaml next to this file.

    prices.yaml fields `input` / `cache_read` / `output` map to the
    `input` / `cached` / `output` keys used internally by this module.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return dict(_PRICING_FALLBACK)
    pricing: dict[str, dict] = {}
    for key, m in (data.get("models") or {}).items():
        if not isinstance(m, dict):
            continue
        pricing[key] = {
            "input": float(m.get("input", 0) or 0),
            "cached": float(m.get("cache_read", 0) or 0),
            "cache_write": float(m.get("cache_write", 0) or 0),
            "output": float(m.get("output", 0) or 0),
            "display": m.get("display") or key,
        }
    return pricing or dict(_PRICING_FALLBACK)


MODEL_PRICING = _load_pricing()


def _norm_model(s: str) -> str:
    """Normalize a model name for fuzzy matching: lowercase, drop non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _pricing_key(model_db: str, model_display: str) -> str | None:
    """Resolve the MODEL_PRICING key for a session.

    The `model` column in sessions.db can lag behind the model actually used
    (e.g. a session started on glm-5-2 but switched to GPT-5.6 Sol mid-session),
    so we trust `model_display` first and only fall back to `model_db`.
    """
    # 1. Match by normalized display name (reflects the model actually used).
    disp_norm = _norm_model(model_display)
    if disp_norm:
        for key in MODEL_PRICING:
            kn = _norm_model(key)
            if kn and kn in disp_norm:
                return key
    # 2. Exact match on the DB model column.
    if model_db in MODEL_PRICING:
        return model_db
    # 3. Fuzzy match on the DB model column.
    db_norm = _norm_model(model_db)
    if db_norm:
        for key in MODEL_PRICING:
            kn = _norm_model(key)
            if kn and (kn in db_norm or db_norm in kn):
                return key
    return None


def _get_pricing(model_db: str, model_display: str) -> dict | None:
    key = _pricing_key(model_db, model_display)
    return MODEL_PRICING[key] if key else None


def _estimate_cost(s: dict) -> float:
    """Estimate USD cost from token counts and model pricing."""
    pricing = _get_pricing(s["model"], s.get("model_display", ""))
    if not pricing:
        return 0.0
    cost = (
        s["input_tokens"] / 1_000_000 * pricing["input"]
        + s["cache_read_tokens"] / 1_000_000 * pricing["cached"]
        + s["cache_creation_tokens"] / 1_000_000 * pricing["cache_write"]
        + s["output_tokens"] / 1_000_000 * pricing["output"]
    )
    return round(cost, 4)


def _pricing_info(model_db: str, model_display: str) -> dict | None:
    pricing = _get_pricing(model_db, model_display)
    if not pricing:
        return None
    return {"input": pricing["input"], "cached": pricing["cached"],
            "cache_write": pricing["cache_write"], "output": pricing["output"]}


# --------------------------------------------------------------------------- #
#  DB helpers
# --------------------------------------------------------------------------- #
def _extract_dims(metadata_json: str | None) -> dict:
    """Parse the session metadata JSON. Newer CLI versions only keep the
    credit/ACU totals here; token usage moved to per-message metrics in
    message_nodes (see _fetch_usage_stats)."""
    result = {"total_credit_cost": 0, "total_acu_cost": 0.0}
    if not metadata_json:
        return result
    try:
        m = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        return result
    result["total_credit_cost"] = m.get("total_credit_cost", 0) or 0
    result["total_acu_cost"] = m.get("total_acu_cost", 0.0) or 0.0
    return result


def _fetch_usage_stats(conn, session_ids: list[str]) -> dict[str, dict]:
    """Sum per-message token metrics (chat_message.metadata.metrics) per session.

    The message tree repeats shared messages across branches, so rows are
    deduplicated by message_id before summing.
    """
    if not session_ids:
        return {}
    ph = ",".join("?" * len(session_ids))
    rows = conn.execute(
        f"""
        SELECT session_id, COUNT(*), SUM(inp), SUM(outp), SUM(cr), SUM(cc)
        FROM (
          SELECT session_id, mid,
                 MAX(inp) inp, MAX(outp) outp, MAX(cr) cr, MAX(cc) cc
          FROM (
            SELECT session_id,
                   json_extract(chat_message,'$.message_id') mid,
                   json_extract(chat_message,'$.metadata.metrics.input_tokens') inp,
                   json_extract(chat_message,'$.metadata.metrics.output_tokens') outp,
                   json_extract(chat_message,'$.metadata.metrics.cache_read_tokens') cr,
                   json_extract(chat_message,'$.metadata.metrics.cache_creation_tokens') cc
            FROM message_nodes
            WHERE session_id IN ({ph})
              AND chat_message LIKE '%"input_tokens"%'
          )
          WHERE mid IS NOT NULL
            AND (inp IS NOT NULL OR outp IS NOT NULL
                 OR cr IS NOT NULL OR cc IS NOT NULL)
          GROUP BY session_id, mid
        )
        GROUP BY session_id
        """,
        session_ids,
    ).fetchall()
    return {
        sid: {
            "agent_messages": msgs,
            "input_tokens": int(inp or 0),
            "output_tokens": int(outp or 0),
            "cache_read_tokens": int(cr or 0),
            "cache_creation_tokens": int(cc or 0),
        }
        for sid, msgs, inp, outp, cr, cc in rows
    }


def _fetch_latest_models(conn, session_ids: list[str]) -> dict[str, str]:
    """Latest 'model' response dimension (display name) per session."""
    if not session_ids:
        return {}
    ph = ",".join("?" * len(session_ids))
    rows = conn.execute(
        f"""
        SELECT session_id, model_name FROM (
          SELECT mn.session_id,
                 json_extract(d.value,'$.kind.Metric.value') model_name,
                 ROW_NUMBER() OVER (PARTITION BY mn.session_id
                                    ORDER BY mn.node_id DESC) rn
          FROM message_nodes mn,
               json_each(mn.chat_message,'$.metadata.response_dimensions') d
          WHERE mn.session_id IN ({ph})
            AND json_extract(d.value,'$.uid') = 'model'
            AND COALESCE(json_extract(mn.chat_message,'$.metadata.generation_model'),'')
                NOT IN ('compactor','summarizer')
        ) WHERE rn = 1
        """,
        session_ids,
    ).fetchall()
    return {sid: name for sid, name in rows if name}


def _get_conn(db_path: str | None = None):
    path = db_path or DB_PATH
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _fetch_sessions(days: int | None = None, db_path: str | None = None) -> list[dict]:
    conn = _get_conn(db_path)
    conn.row_factory = sqlite3.Row
    if days is not None:
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        rows = conn.execute(
            "SELECT id, working_directory, model, created_at, last_activity_at, "
            "title, metadata FROM sessions WHERE last_activity_at >= ? "
            "ORDER BY last_activity_at DESC",
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, working_directory, model, created_at, last_activity_at, "
            "title, metadata FROM sessions ORDER BY last_activity_at DESC"
        ).fetchall()
    session_ids = [r["id"] for r in rows]
    usage = _fetch_usage_stats(conn, session_ids)
    models = _fetch_latest_models(conn, session_ids)
    conn.close()
    sessions = []
    for r in rows:
        dims = _extract_dims(r["metadata"])
        u = usage.get(r["id"], {})
        inp = u.get("input_tokens", 0)
        outp = u.get("output_tokens", 0)
        cached = u.get("cache_read_tokens", 0) + u.get("cache_creation_tokens", 0)
        model_display = models.get(r["id"]) or r["model"]
        model_group = (_pricing_key(r["model"], model_display)
                       or _norm_model(r["model"] or model_display)
                       or r["model"] or model_display)
        sessions.append({
            "id": r["id"],
            "title": r["title"] or "",
            "model": r["model"],
            "model_display": model_display,
            "model_group": model_group,
            "working_directory": r["working_directory"],
            "created_at": r["created_at"],
            "created_at_str": datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M"),
            "last_activity_at": r["last_activity_at"],
            "last_activity_at_str": datetime.fromtimestamp(r["last_activity_at"]).strftime("%Y-%m-%d %H:%M"),
            "input_tokens": inp,
            "output_tokens": outp,
            "cached_input_tokens": cached,
            "total_tokens": inp + outp + cached,
            "agent_messages": u.get("agent_messages", 0),
            "total_credit_cost": dims["total_credit_cost"],
            "total_acu_cost": dims["total_acu_cost"],
            "estimated_cost": _estimate_cost({
                "model": r["model"],
                "model_display": model_display,
                "input_tokens": inp,
                "output_tokens": outp,
                "cache_read_tokens": u.get("cache_read_tokens", 0),
                "cache_creation_tokens": u.get("cache_creation_tokens", 0),
            }),
        })
    return sessions


def _aggregate(sessions: list[dict]) -> dict:
    return {
        "session_count": len(sessions),
        "agent_messages": sum(s["agent_messages"] for s in sessions),
        "input_tokens": sum(s["input_tokens"] for s in sessions),
        "output_tokens": sum(s["output_tokens"] for s in sessions),
        "cached_input_tokens": sum(s["cached_input_tokens"] for s in sessions),
        "total_tokens": sum(s["total_tokens"] for s in sessions),
        "total_credit_cost": sum(s["total_credit_cost"] for s in sessions),
        "total_acu_cost": sum(s["total_acu_cost"] for s in sessions),
        "estimated_cost": round(sum(s["estimated_cost"] for s in sessions), 2),
    }


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sessions")
def api_sessions():
    days = request.args.get("days", type=int)
    db_path = request.args.get("db")
    try:
        sessions = _fetch_sessions(days=days, db_path=db_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"aggregate": _aggregate(sessions), "sessions": sessions})


@app.route("/api/by-model")
def api_by_model():
    days = request.args.get("days", type=int)
    db_path = request.args.get("db")
    try:
        sessions = _fetch_sessions(days=days, db_path=db_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    by_model: dict[str, dict] = {}
    for s in sessions:
        # Group by canonical model id (pricing key) so the same model doesn't
        # split into "GLM-5.2 High" / "glm-5-2" rows when the display name
        # is missing from older messages.
        pkey = _pricing_key(s["model"], s["model_display"])
        key = s["model_group"]
        if key not in by_model:
            display = MODEL_PRICING[key].get("display", key) if pkey else (s["model_display"] or key)
            by_model[key] = {
                "model": display,
                "model_group": key,
                "model_db": s["model"],
                "session_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "total_tokens": 0,
                "agent_messages": 0,
                "estimated_cost": 0.0,
                "pricing": _pricing_info(s["model"], s["model_display"]),
            }
        m = by_model[key]
        m["session_count"] += 1
        m["input_tokens"] += s["input_tokens"]
        m["output_tokens"] += s["output_tokens"]
        m["cached_input_tokens"] += s["cached_input_tokens"]
        m["total_tokens"] += s["total_tokens"]
        m["agent_messages"] += s["agent_messages"]
        m["estimated_cost"] += s["estimated_cost"]
    for m in by_model.values():
        m["estimated_cost"] = round(m["estimated_cost"], 2)
    return jsonify(sorted(by_model.values(), key=lambda x: x["estimated_cost"], reverse=True))


@app.route("/api/by-day")
def api_by_day():
    days = request.args.get("days", type=int)
    db_path = request.args.get("db")
    try:
        sessions = _fetch_sessions(days=days, db_path=db_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    # When the range is 1 day, aggregate by hour for a meaningful trend.
    by_hour = days is not None and days <= 1
    by_day: dict[str, dict] = {}
    for s in sessions:
        if by_hour:
            key = datetime.fromtimestamp(s["last_activity_at"]).strftime("%Y-%m-%d %H:00")
        else:
            key = datetime.fromtimestamp(s["last_activity_at"]).strftime("%Y-%m-%d")
        if key not in by_day:
            by_day[key] = {
                "date": key,
                "session_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "total_tokens": 0,
                "agent_messages": 0,
                "estimated_cost": 0.0,
            }
        d = by_day[key]
        d["session_count"] += 1
        d["input_tokens"] += s["input_tokens"]
        d["output_tokens"] += s["output_tokens"]
        d["cached_input_tokens"] += s["cached_input_tokens"]
        d["total_tokens"] += s["total_tokens"]
        d["agent_messages"] += s["agent_messages"]
        d["estimated_cost"] += s["estimated_cost"]
    for d in by_day.values():
        d["estimated_cost"] = round(d["estimated_cost"], 2)
    return jsonify(sorted(by_day.values(), key=lambda x: x["date"]))


@app.route("/api/session/<session_id>")
def api_session_detail(session_id):
    db_path = request.args.get("db")
    try:
        conn = _get_conn(db_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, working_directory, model, created_at, last_activity_at, title, metadata "
        "FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    u = _fetch_usage_stats(conn, [session_id]).get(session_id, {})
    models = _fetch_latest_models(conn, [session_id])
    inp = u.get("input_tokens", 0)
    outp = u.get("output_tokens", 0)
    cached = u.get("cache_read_tokens", 0) + u.get("cache_creation_tokens", 0)
    session_info = {
        "id": row["id"],
        "title": row["title"] or "",
        "model": row["model"],
        "model_display": models.get(session_id) or row["model"],
        "working_directory": row["working_directory"],
        "created_at_str": datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M"),
        "last_activity_at_str": datetime.fromtimestamp(row["last_activity_at"]).strftime("%Y-%m-%d %H:%M"),
        "input_tokens": inp,
        "output_tokens": outp,
        "cached_input_tokens": cached,
        "total_tokens": inp + outp + cached,
        "agent_messages": u.get("agent_messages", 0),
    }
    nodes = conn.execute(
        "SELECT node_id, parent_node_id, chat_message, created_at "
        "FROM message_nodes WHERE session_id = ? ORDER BY node_id",
        (session_id,),
    ).fetchall()
    conn.close()
    messages = []
    for n in nodes:
        try:
            msg = json.loads(n["chat_message"])
        except (json.JSONDecodeError, TypeError):
            continue
        messages.append({
            "node_id": n["node_id"],
            "parent_node_id": n["parent_node_id"],
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "tool_call_id": msg.get("tool_call_id", ""),
            "created_at_str": datetime.fromtimestamp(n["created_at"]).strftime("%H:%M:%S"),
            "msg_size": len(n["chat_message"]),
        })
    return jsonify({"session": session_info, "messages": messages})


@app.route("/session/<session_id>")
def session_detail_page(session_id):
    return render_template("session_detail.html", session_id=session_id)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5117, debug=True)

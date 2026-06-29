#!/usr/bin/env python3
"""
Integration tests for the OpenCode -> OpenClaw-RL bridge.

These simulate exactly the HTTP requests that the `opencode-rl-headers` plugin
causes OpenCode to send to the OpenClaw-RL proxy (openclaw_*_api_server.py),
and verify the proxy accepts them, returns valid OpenAI-compatible responses,
echoes the session id, and populates the training record file.

Usage:
    python3 test_integration.py

Environment overrides:
    RL_PROXY_URL   (default http://127.0.0.1:30000)
    RL_API_KEY     (default sk-1234)
    RL_MODEL       (default glm-4.7-flash)
    RL_RECORD_FILE (default /workspace/OpenClaw-RL/openclaw-combine/results/glm4.7_flash_record.jsonl)
"""
import json
import os
import time

import httpx

PROXY_BASE = os.getenv("RL_PROXY_URL", "http://127.0.0.1:30000").rstrip("/")
CHAT_URL = f"{PROXY_BASE}/v1/chat/completions"
HEALTH_URL = f"{PROXY_BASE}/healthz"
API_KEY = os.getenv("RL_API_KEY", "sk-1234")
MODEL = os.getenv("RL_MODEL", "glm-4.7-flash")
RECORD_FILE = os.getenv(
    "RL_RECORD_FILE",
    "/workspace/OpenClaw-RL/openclaw-combine/results/glm4.7_flash_record.jsonl",
)

_GREEN, _RED, _YELLOW, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

_results = {"passed": 0, "failed": 0, "skipped": 0}


def _ok(msg):
    print(f"  {_GREEN}PASS{_RESET} {msg}")
    _results["passed"] += 1


def _fail(msg):
    print(f"  {_RED}FAIL{_RESET} {msg}")
    _results["failed"] += 1


def _skip(msg):
    print(f"  {_YELLOW}SKIP{_RESET} {msg}")
    _results["skipped"] += 1


def make_request(session_id, turn_type, messages, *, session_done=False,
                 tools=None, max_tokens=200, timeout=120):
    """Replicate the headers the opencode plugin injects via chat.headers."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "X-Session-Id": session_id,
        "X-Turn-Type": turn_type,
    }
    if session_done:
        headers["X-Session-Done"] = "true"

    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    return httpx.post(CHAT_URL, headers=headers, json=body, timeout=timeout)


def _read_records():
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            out = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return out
    except FileNotFoundError:
        return None


def _count_records(session_prefix=None):
    recs = _read_records()
    if recs is None:
        return None
    if session_prefix is None:
        return len(recs)
    return sum(1 for r in recs if str(r.get("session_id", "")).startswith(session_prefix))


# --------------------------------------------------------------------------- #
def test_health():
    print("=== Test 0: proxy health ===")
    try:
        r = httpx.get(HEALTH_URL, timeout=10)
        if r.status_code == 200 and r.json().get("ok"):
            _ok(f"proxy healthy at {HEALTH_URL}")
            return True
        _fail(f"unexpected health response: {r.status_code} {r.text}")
    except Exception as e:
        _fail(f"cannot reach proxy: {e}")
    return False


def test_single_turn():
    print("\n=== Test 1: single main turn ===")
    sid = f"oc-test-single-{int(time.time())}"
    try:
        r = make_request(sid, "main", [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ])
    except Exception as e:
        _fail(f"request raised: {e}")
        return None
    if r.status_code != 200:
        _fail(f"status={r.status_code} body={r.text[:300]}")
        return None
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        _fail(f"empty content: {json.dumps(data)[:300]}")
        return None
    _ok(f"got response ({len(content)} chars): {content[:60]!r}")
    if data.get("session_id") == sid:
        _ok("response echoed X-Session-Id correctly")
    else:
        _fail(f"session_id mismatch: got {data.get('session_id')!r} want {sid!r}")
    return sid


def test_multi_turn_with_tool_calls():
    print("\n=== Test 2: multi-turn with tool_calls in history ===")
    sid = f"oc-test-tools-{int(time.time())}"
    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Read the file /etc/hostname"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "cat /etc/hostname"}),
            },
        }]},
        {"role": "tool", "tool_call_id": "call_001", "content": "my-server\n"},
        {"role": "user", "content": "Good. Now just say DONE."},
    ]
    try:
        r = make_request(sid, "main", messages)
    except Exception as e:
        _fail(f"request raised: {e}")
        return None
    if r.status_code != 200:
        _fail(f"status={r.status_code} body={r.text[:300]}")
        return None
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if content is not None:
        _ok(f"tool-call history accepted, response: {str(content)[:60]!r}")
    else:
        _fail(f"no content field: {json.dumps(data)[:300]}")
    return sid


def test_tool_definitions_passthrough():
    print("\n=== Test 3: request carrying tool definitions ===")
    sid = f"oc-test-tooldef-{int(time.time())}"
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        r = make_request(sid, "main", [
            {"role": "user", "content": "What's the weather in Paris?"},
        ], tools=tools, max_tokens=256)
    except Exception as e:
        _fail(f"request raised: {e}")
        return None
    if r.status_code != 200:
        _fail(f"status={r.status_code} body={r.text[:300]}")
        return None
    msg = r.json().get("choices", [{}])[0].get("message", {})
    if msg.get("tool_calls"):
        _ok(f"model emitted tool_calls: {json.dumps(msg['tool_calls'])[:80]}")
    else:
        _ok("request with tools accepted (model chose text reply)")
    return sid


def test_side_turn():
    print("\n=== Test 4: side turn (no training data expected) ===")
    sid = f"oc-test-side-{int(time.time())}"
    before = _count_records(sid)
    try:
        r = make_request(sid, "side", [
            {"role": "user", "content": "background housekeeping task"},
        ])
    except Exception as e:
        _fail(f"request raised: {e}")
        return None
    if r.status_code != 200:
        _fail(f"status={r.status_code} body={r.text[:300]}")
        return None
    _ok("side request accepted by proxy")
    if before is not None:
        time.sleep(1)
        after = _count_records(sid)
        if after == 0:
            _ok("side turn produced no training record (as expected)")
        else:
            _fail(f"side turn unexpectedly produced {after} record(s)")
    else:
        _skip("record file unavailable; cannot assert side-turn exclusion")
    return sid


def test_session_done_flow():
    print("\n=== Test 5: full session with done signal ===")
    sid = f"oc-test-done-{int(time.time())}"
    try:
        r1 = make_request(sid, "main", [
            {"role": "user", "content": "Hello, introduce yourself in one line."},
        ])
        if r1.status_code != 200:
            _fail(f"turn1 status={r1.status_code} body={r1.text[:200]}")
            return None
        turn1_reply = r1.json()["choices"][0]["message"].get("content") or ""

        r2 = make_request(sid, "main", [
            {"role": "user", "content": "Hello, introduce yourself in one line."},
            {"role": "assistant", "content": turn1_reply},
            {"role": "user", "content": "Thanks, goodbye."},
        ])
        if r2.status_code != 200:
            _fail(f"turn2 status={r2.status_code} body={r2.text[:200]}")
            return None

        # Mimic the plugin's session.idle -> X-Session-Done request.
        # Sent as a "side" turn with minimal non-empty content (SGLang rejects
        # empty content). Session-done cleanup runs regardless of turn type.
        r3 = make_request(sid, "side",
                          [{"role": "user", "content": "."}],
                          session_done=True, max_tokens=1)
        if r3.status_code != 200:
            _fail(f"session-done status={r3.status_code} body={r3.text[:200]}")
            return None
    except Exception as e:
        _fail(f"request raised: {e}")
        return None
    _ok("two main turns + session-done signal all accepted")
    return sid


def test_record_file(sids):
    print("\n=== Test 6: training record file populated ===")
    recs = _read_records()
    if recs is None:
        _skip(f"record file not found: {RECORD_FILE} (OPENCLAW_RECORD_ENABLED off?)")
        return
    time.sleep(1)
    recs = _read_records() or []
    _ok(f"record file readable with {len(recs)} total entries")
    test_sids = {s for s in sids if s}
    matched = [r for r in recs if r.get("session_id") in test_sids]
    if matched:
        _ok(f"{len(matched)} record(s) from this test run found")
        sample = matched[-1]
        for key in ("session_id", "turn", "prompt_text", "response_text"):
            if key in sample:
                _ok(f"record has field '{key}'")
            else:
                _fail(f"record missing field '{key}'")
        print(f"    last record: session={sample.get('session_id')} "
              f"turn={sample.get('turn')} "
              f"prompt_len={len(sample.get('prompt_text',''))} "
              f"resp_len={len(sample.get('response_text',''))}")
    else:
        _skip("no records matched this run's session ids yet "
              "(PRM/async submission may still be in flight)")


def main():
    print("OpenCode -> OpenClaw-RL Integration Tests")
    print(f"proxy={PROXY_BASE} model={MODEL}")
    print("=" * 55)

    if not test_health():
        print(f"\n{_RED}Proxy not reachable - aborting.{_RESET}")
        return 1

    sids = []
    sids.append(test_single_turn())
    sids.append(test_multi_turn_with_tool_calls())
    sids.append(test_tool_definitions_passthrough())
    sids.append(test_side_turn())
    sids.append(test_session_done_flow())
    test_record_file(sids)

    print("\n" + "=" * 55)
    print(f"RESULT: {_GREEN}{_results['passed']} passed{_RESET}, "
          f"{_RED}{_results['failed']} failed{_RESET}, "
          f"{_YELLOW}{_results['skipped']} skipped{_RESET}")
    return 1 if _results["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

# OpenCode RL Headers (OpenCode -> OpenClaw-RL bridge)

OpenCode plugin that feeds OpenCode interaction data into the **OpenClaw-RL**
training pipeline, giving OpenCode the same RL training/inference hook-up that
OpenClaw gets from the `rl-training-headers` extension.

It is the OpenCode-native counterpart of `extensions/rl-training-headers`
(which targets OpenClaw). Same idea, different host application.

## How it works

The OpenClaw-RL proxy (`openclaw_api_server.py`, `openclaw_opd_api_server.py`,
`openclaw_combine_api_server.py`, ...) is an **OpenAI-compatible** server that
sits in front of SGLang. It reads three HTTP headers on every
`/v1/chat/completions` request to segment training data:

| Header | Value | Meaning |
|---|---|---|
| `X-Session-Id` | `<opencode session id>` | groups turns into a trajectory |
| `X-Turn-Type` | `main` \| `side` | `main` = user-facing (trained on); `side` = subtask/housekeeping (skipped) |
| `X-Session-Done` | `true` | session finished -> flush + submit final turn |

Because OpenCode talks to providers over the same OpenAI-compatible wire
format, the bridge needs **no extra proxy server**. It only has to:

1. Point OpenCode's LLM provider at the OpenClaw-RL proxy (`opencode.jsonc`).
2. Inject the three headers on every request (this plugin, via the
   `chat.headers` hook).
3. Signal session end when a session goes idle (this plugin, via the `event`
   hook listening for `session.idle` / `session.deleted`).

```
OpenCode (TUI / run)
  │  chat.headers  -> X-Session-Id, X-Turn-Type
  │  event(idle)   -> X-Session-Done (side turn)
  │  provider baseURL -> OpenClaw-RL proxy
  ▼
OpenClaw-RL proxy (:30000)  ->  SGLang  ->  training queue + record JSONL
```

### Comparison with the OpenClaw extension

| | OpenClaw `rl-training-headers` | OpenCode `opencode-rl-headers` |
|---|---|---|
| Header injection | monkey-patch `globalThis.fetch` in `before_prompt_build` | native `chat.headers` hook |
| `main` vs `side` | from `ctx.trigger` (`heartbeat`/`memory`/`cron` -> side) | from `agent` (`explore` -> side) |
| Session done | gateway lifecycle | `session.idle` / `session.deleted` event |
| Provider wiring | `openclaw.json` `models.providers.*.baseUrl` | `opencode.jsonc` `provider.*.options.baseURL` |

## Install

1. Configure OpenCode (`~/.config/opencode/opencode.jsonc`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "rl-proxy": {
      "api": "openai",
      "name": "RL Training Proxy (GLM-4.7-Flash)",
      "options": {
        "baseURL": "http://127.0.0.1:30000/v1",
        "apiKey": "sk-1234"
      },
      "models": {
        "glm-4.7-flash": {
          "name": "GLM-4.7-Flash (RL Training)",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  },
  "model": "rl-proxy/glm-4.7-flash",
  "plugin": [
    ["/workspace/OpenClaw-RL/extensions/opencode-rl-headers", {
      "bridgeUrl": "http://127.0.0.1:30000/v1/chat/completions",
      "apiKey": "sk-1234",
      "modelName": "glm-4.7-flash"
    }]
  ]
}
```

The `apiKey` / `baseURL` must match the proxy's `SGLANG_API_KEY` and
`HOST`/`PORT`; the model name must match `SERVED_MODEL_NAME`.

2. Select the model in OpenCode (`rl-proxy/glm-4.7-flash`) and start chatting.
   Every request now carries the RL training headers.

## Plugin options

| Option | Default | Description |
|---|---|---|
| `bridgeUrl` | `http://127.0.0.1:30000/v1/chat/completions` | proxy endpoint for the session-done ping |
| `apiKey` | `sk-1234` | bearer token (matches `SGLANG_API_KEY`) |
| `modelName` | `glm-4.7-flash` | model id used in the session-done ping |
| `sessionIdHeader` | `X-Session-Id` | header name override |
| `turnTypeHeader` | `X-Turn-Type` | header name override |
| `sessionDoneHeader` | `X-Session-Done` | header name override |
| `sideAgents` | `["explore"]` | agents classified as `side` turns |

## Testing

`test/test_integration.py` replicates the exact requests the plugin produces
and asserts the proxy accepts them, echoes the session id, excludes `side`
turns, handles the session-done signal, and writes training records.

```bash
python3 test/test_integration.py
```

Environment overrides: `RL_PROXY_URL`, `RL_API_KEY`, `RL_MODEL`,
`RL_RECORD_FILE`.

End-to-end check driving the real binary:

```bash
opencode run --print-logs -m rl-proxy/glm-4.7-flash "say hello"
# then grep the proxy log for the opencode session id (ses_...) and
# confirm: has_x_session_id=True has_x_turn_type=True
```

## Notes

- SGLang rejects empty message content, so the session-done ping sends a
  minimal `"."` prompt with `max_tokens=1`, tagged as a `side` turn so it is
  never recorded as a training sample. Session-done cleanup in the proxy runs
  regardless of turn type, so the last real turn is still flushed.
- In OPD/combine modes a turn is only kept if it has a *next_state* (a
  following user/tool message used for PRM scoring). Single-turn sessions are
  therefore dropped by design; multi-turn sessions yield training samples.

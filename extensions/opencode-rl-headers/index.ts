import type { PluginModule } from "@opencode-ai/plugin";

/**
 * OpenCode plugin that bridges OpenCode interaction data into the OpenClaw-RL
 * training pipeline.
 *
 * It mirrors the behaviour of the OpenClaw `rl-training-headers` extension but
 * uses OpenCode's native `chat.headers` hook (instead of monkey-patching
 * `globalThis.fetch`) to inject the RL session metadata headers consumed by
 * the OpenClaw-RL proxy (`openclaw_api_server.py` / `openclaw_combine_api_server.py`):
 *
 *   - X-Session-Id   : the active OpenCode session id (groups training turns)
 *   - X-Turn-Type    : "main" (user-facing) | "side" (subtask / housekeeping)
 *   - X-Session-Done : "true" when a session is considered finished
 *
 * The OpenClaw-RL proxy must be configured as the LLM provider in
 * `opencode.jsonc` (api: "openai", baseURL pointing at the proxy). The proxy
 * forwards requests to SGLang, captures logprobs, tokenises prompt+response
 * and submits training samples - exactly as it does for OpenClaw.
 */

type RlOptions = {
  bridgeUrl?: string;
  apiKey?: string;
  modelName?: string;
  sessionIdHeader?: string;
  turnTypeHeader?: string;
  sessionDoneHeader?: string;
  sideAgents?: string[];
};

// OpenCode agents treated as "side" turns (no user-facing training reward).
// `explore` is OpenCode's read-only research subagent, analogous to OpenClaw's
// heartbeat / memory / cron housekeeping turns.
const DEFAULT_SIDE_AGENTS = ["explore"];

export default {
  id: "opencode-rl-headers",
  server: async (input, options) => {
    const opts = (options ?? {}) as RlOptions;

    const sessionIdHeader = opts.sessionIdHeader ?? "X-Session-Id";
    const turnTypeHeader = opts.turnTypeHeader ?? "X-Turn-Type";
    const sessionDoneHeader = opts.sessionDoneHeader ?? "X-Session-Done";
    const bridgeUrl = opts.bridgeUrl ?? "http://127.0.0.1:30000/v1/chat/completions";
    const apiKey = opts.apiKey ?? "sk-1234";
    const modelName = opts.modelName ?? "glm-4.7-flash";
    const sideAgents = new Set(opts.sideAgents ?? DEFAULT_SIDE_AGENTS);

    // Tracks the most recent session that issued a "main" request, so we know
    // which session to mark done when it goes idle.
    let lastMainSessionId: string | null = null;
    // Sessions for which we have already fired the done signal (avoid dupes).
    const doneSessions = new Set<string>();

    const log = (msg: string) => {
      try {
        input?.client?.log?.({
          body: { service: "opencode-rl-headers", level: "info", message: msg },
        });
      } catch {
        // best-effort logging only
      }
    };

    log(`activated → bridge=${bridgeUrl} model=${modelName}`);

    return {
      /**
       * Inject RL training headers into every outgoing LLM request.
       */
      "chat.headers": async (hookInput, output) => {
        const sessionId = hookInput.sessionID ?? "";
        const agent = hookInput.agent ?? "";
        const turnType = sideAgents.has(agent) ? "side" : "main";

        output.headers[sessionIdHeader] = sessionId;
        output.headers[turnTypeHeader] = turnType;

        if (turnType === "main" && sessionId) {
          lastMainSessionId = sessionId;
          // A fresh main request means the session is active again.
          doneSessions.delete(sessionId);
        }
      },

      /**
       * Watch session lifecycle. When a session becomes idle we notify the
       * OpenClaw-RL proxy that the session is finished by sending a minimal
       * request carrying X-Session-Done. The proxy flushes any pending record
       * and submits the final turn's training sample.
       */
      "event": async ({ event }) => {
        const isIdle =
          event?.type === "session.idle" ||
          event?.type === "session.deleted";
        if (!isIdle) return;

        // Resolve the session id from the event payload, falling back to the
        // last active main session. `session.idle` carries `properties.sessionID`
        // while `session.deleted` carries `properties.info.id`.
        const props = (event as any)?.properties ?? {};
        const sessionId: string =
          props.sessionID || props.sessionId || props.info?.id || lastMainSessionId || "";

        if (!sessionId || doneSessions.has(sessionId)) return;
        doneSessions.add(sessionId);
        if (sessionId === lastMainSessionId) lastMainSessionId = null;

        try {
          await fetch(bridgeUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${apiKey}`,
              [sessionIdHeader]: sessionId,
              // Mark as "side" so the proxy does NOT record this synthetic
              // ping as a training turn. Session-done cleanup runs regardless
              // of turn type, so the last real turn is still flushed.
              [turnTypeHeader]: "side",
              [sessionDoneHeader]: "true",
            },
            body: JSON.stringify({
              // SGLang rejects empty content, so send a minimal non-empty
              // prompt with max_tokens=1 to keep the round-trip cheap.
              model: modelName,
              messages: [{ role: "user", content: "." }],
              max_tokens: 1,
              session_id: sessionId,
              turn_type: "side",
              session_done: true,
            }),
          });
          log(`session done signalled: ${sessionId}`);
        } catch (e) {
          // Non-fatal: never break OpenCode if the proxy is unreachable.
          log(`session done signal failed for ${sessionId}: ${String(e)}`);
        }
      },
    };
  },
} satisfies PluginModule;

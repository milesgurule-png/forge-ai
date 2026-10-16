#!/usr/bin/env python3
# Copyright 2026 Miles Gurule / Forge Labs
# SPDX-License-Identifier: Apache-2.0
"""Forge AI - minimal CLI chat with a local Ollama server.

Uses only the Python standard library. Talks to Ollama's HTTP API
(POST /api/chat, streaming newline-delimited JSON).

Usage:
    python3 chat.py                      # default model qwen3.5:9b
    python3 chat.py --model gemma4:e4b
    python3 chat.py --think              # let the model "think" first
    FORGE_MODEL=phi4-mini python3 chat.py

In-chat commands: /reset (clear history), /exit or Ctrl-D (quit).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("FORGE_MODEL", "qwen3.5:9b")
DEFAULT_SYSTEM = (
    "You are Forge AI, an open-source assistant built by Forge Labs for Miles. "
    "Be concise, accurate, and say when you are unsure."
)

NOT_RUNNING_MSG = """\
Error: can't reach Ollama at {host}

Forge AI needs a local Ollama server. To fix:
  1. Install Ollama:   brew install ollama   (or download from https://ollama.com/download)
  2. Start it:         ollama serve          (or open the Ollama app)
  3. Pull the model:   ollama pull {model}
Then run this script again. Set OLLAMA_HOST if Ollama runs elsewhere.
"""


def normalize_host(host):
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def check_server(host, model):
    """Return True if Ollama answers; print a helpful error and return False otherwise."""
    try:
        with urllib.request.urlopen(host + "/api/version", timeout=3) as resp:
            json.loads(resp.read().decode("utf-8"))
        return True
    except (urllib.error.URLError, ConnectionError, OSError, ValueError):
        sys.stderr.write(NOT_RUNNING_MSG.format(host=host, model=model))
        return False


def stream_chat(host, model, messages, think, show_thinking):
    """Send the conversation to Ollama and stream the reply to stdout.

    Returns the assistant's full reply text (without thinking).
    """
    payload = {"model": model, "messages": messages, "stream": True, "think": think}
    req = urllib.request.Request(
        host + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    reply_parts = []
    in_thinking = False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            thinking = msg.get("thinking")
            if thinking and show_thinking:
                if not in_thinking:
                    sys.stdout.write("\033[2m[thinking] ")
                    in_thinking = True
                sys.stdout.write(thinking)
                sys.stdout.flush()
            content = msg.get("content", "")
            if content:
                if in_thinking:
                    sys.stdout.write("\033[0m\n\n")
                    in_thinking = False
                reply_parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
            if chunk.get("done"):
                break
    if in_thinking:
        sys.stdout.write("\033[0m")
    sys.stdout.write("\n")
    return "".join(reply_parts)


def explain_http_error(err, model):
    try:
        detail = json.loads(err.read().decode("utf-8")).get("error", "")
    except Exception:
        detail = ""
    if err.code == 404 or "not found" in detail.lower():
        return "Model '{0}' isn't downloaded. Run: ollama pull {0}".format(model)
    if "think" in detail.lower():
        return "{} (this model may not support thinking; try without --think)".format(detail)
    return "Ollama returned HTTP {}: {}".format(err.code, detail or err.reason)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Forge AI: chat with a local Ollama model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag (default: %(default)s)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL (default: %(default)s)")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--think", action="store_true", help="Enable the model's thinking mode (slower)")
    parser.add_argument("--show-thinking", action="store_true", help="Print thinking text (implies --think)")
    args = parser.parse_args(argv)

    host = normalize_host(args.host)
    think = args.think or args.show_thinking
    if not check_server(host, args.model):
        return 1

    history = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(history)
    print("Forge AI - model {} via {}. Type /reset to clear, /exit to quit.".format(args.model, host))

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if user == "/reset":
            messages = list(history)
            print("(history cleared)")
            continue

        messages.append({"role": "user", "content": user})
        sys.stdout.write("forge> ")
        sys.stdout.flush()
        try:
            reply = stream_chat(host, args.model, messages, think, args.show_thinking)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            messages.pop()
            continue
        except urllib.error.HTTPError as err:
            print("\nError: " + explain_http_error(err, args.model))
            messages.pop()
            continue
        except (urllib.error.URLError, ConnectionError, OSError):
            print()
            sys.stderr.write(NOT_RUNNING_MSG.format(host=host, model=args.model))
            messages.pop()
            continue
        except (RuntimeError, ValueError) as err:
            print("\nError from Ollama: {}".format(err))
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())

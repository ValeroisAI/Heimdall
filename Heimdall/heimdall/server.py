"""
OpenAI uyumlu HTTP API (SaaS için).

    python -m heimdall.server --model models/heimdall-temel --port 8000

Uç noktalar:
    POST /v1/completions, POST /v1/chat/completions   (stream=true ile SSE)
    GET  /v1/models, /health, /metrics (Prometheus)
Güvenlik ve işletme:
    - API anahtarı: `Authorization: Bearer <anahtar>`. Anahtarlar HEIMDALL_API_KEYS (virgülle) ya da
      --keys-file (satır başına düz anahtar veya "sha256:<hex>"). Bellekte yalnız SHA-256 özetleri tutulur.
    - Anahtar başına dakikalık istek sınırı ve eşzamanlı istek sınırı (429).
    - Kullanım kaydı (faturalama): her istek için JSONL satırı.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Union

# The allocator reads this at the first GPU allocation, so it must be set before torch is imported (as in train.py):
# growing KV caches of different sizes otherwise fragment memory on long prompts.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import Depends, FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import kernels
from .engine import Engine, Request
from .io import load_model, tokenizer_path

log = logging.getLogger("heimdall.server")


@dataclass
class Settings:
    model_name: str = "heimdall"
    api_key_hashes: dict = field(default_factory=dict)   # sha256 hex → kısa kimlik
    rpm: int = 120                                        # anahtar başına dakikalık istek
    max_concurrent: int = 8                               # anahtar başına eşzamanlı istek
    max_tokens_cap: int = 2048
    usage_log: Optional[str] = None
    chat_template: str = "{system}{history}<|asistan|>\n"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_keys(env_value: Optional[str], keys_file: Optional[str]) -> dict:
    hashes = []
    for k in (env_value or "").split(","):
        if k.strip():
            hashes.append(hash_key(k.strip()))
    if keys_file:
        for line in open(keys_file, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                hashes.append(line[7:] if line.startswith("sha256:") else hash_key(line))
    return {h: h[:10] for h in hashes}


# --------------------------------------------------------------------------- istek şemaları
class CompletionBody(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = 1


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(CompletionBody):
    messages: List[ChatMessage]


# --------------------------------------------------------------------------- yardımcılar
class Detokenizer:
    """Artımlı çözümleme: yarım UTF-8 karakterlerini bekletir (vLLM tarzı önek penceresi)."""

    def __init__(self, tok) -> None:
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def push(self, t: int) -> str:
        self.ids.append(t)
        before = self.tok.decode(self.ids[self.prefix:self.read])
        after = self.tok.decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return after[len(before):]
        return ""


class StopMatcher:
    """Durdurma dizilerini akışta yakalar; bir durdurma dizisinin başı olabilecek son ek bekletilir."""

    def __init__(self, stops: List[str]) -> None:
        self.stops, self.buf = [s for s in stops if s], ""

    def feed(self, text: str):
        self.buf += text
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0:
                out, self.buf = self.buf[:i], ""
                return out, True
        hold = 0
        for s in self.stops:
            for n in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:n]):
                    hold = max(hold, n)
                    break
        out = self.buf[:len(self.buf) - hold]
        self.buf = self.buf[len(self.buf) - hold:]
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


class RateLimiter:
    def __init__(self, rpm: int, max_concurrent: int) -> None:
        self.rpm, self.max_concurrent = rpm, max_concurrent
        self.buckets, self.active, self.lock = {}, {}, threading.Lock()

    def acquire(self, key: str) -> None:
        now = time.time()
        with self.lock:
            tokens, last = self.buckets.get(key, (float(self.rpm), now))
            tokens = min(float(self.rpm), tokens + (now - last) * self.rpm / 60.0)
            if tokens < 1.0:
                raise HTTPException(429, "istek sınırı aşıldı (dakikalık)")
            if self.active.get(key, 0) >= self.max_concurrent:
                raise HTTPException(429, "eşzamanlı istek sınırı aşıldı")
            self.buckets[key] = (tokens - 1.0, now)
            self.active[key] = self.active.get(key, 0) + 1

    def release(self, key: str) -> None:
        with self.lock:
            self.active[key] = max(0, self.active.get(key, 1) - 1)


class Metrics:
    def __init__(self) -> None:
        self.c = {"requests_total": 0, "requests_failed_total": 0, "prompt_tokens_total": 0,
                  "completion_tokens_total": 0, "latency_seconds_sum": 0.0, "ttft_seconds_sum": 0.0}
        self.lock = threading.Lock()

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                self.c[k] += v

    def render(self, engine: Engine) -> str:
        lines = [f"heimdall_{k} {v}" for k, v in self.c.items()]
        lines += [f"heimdall_active_sequences {len(engine.active)}", f"heimdall_queue_depth {engine.queued}",
                  f"heimdall_engine_errors_total {engine.stats['errors']}"]
        if torch.cuda.is_available():
            lines.append(f"heimdall_gpu_memory_bytes {torch.cuda.memory_allocated()}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- uygulama
def create_app(engine: Engine, tokenizer, settings: Settings) -> FastAPI:
    app = FastAPI(title="Heimdall API", version="1.0")
    limiter, metrics = RateLimiter(settings.rpm, settings.max_concurrent), Metrics()
    usage_lock = threading.Lock()
    eos = tokenizer.token_to_id("<eos>")
    if not settings.api_key_hashes:
        log.warning("API anahtarı tanımlı değil: kimlik doğrulama KAPALI (yalnız geliştirme için)")

    def auth(request: HTTPRequest) -> str:
        if not settings.api_key_hashes:
            return "anon"
        header = request.headers.get("authorization", "")
        key = header[7:].strip() if header.lower().startswith("bearer ") else ""
        kid = settings.api_key_hashes.get(hash_key(key)) if key else None
        if kid is None:
            raise HTTPException(401, "geçersiz ya da eksik API anahtarı")
        return kid

    def record(kid, endpoint, n_prompt, n_out, finish, t0, t_first) -> None:
        latency = time.time() - t0
        metrics.add(requests_total=1, prompt_tokens_total=n_prompt, completion_tokens_total=n_out,
                    latency_seconds_sum=latency, ttft_seconds_sum=(t_first or time.time()) - t0,
                    requests_failed_total=int(finish == "error"))
        if settings.usage_log:
            row = {"ts": round(t0, 3), "key": kid, "endpoint": endpoint, "model": settings.model_name,
                   "prompt_tokens": n_prompt, "completion_tokens": n_out, "finish": finish,
                   "latency_ms": round(latency * 1000), "ttft_ms": round(((t_first or time.time()) - t0) * 1000)}
            with usage_lock, open(settings.usage_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    async def generate(ids: List[int], body: CompletionBody):
        """(metin parçası, bitiş nedeni | None) üretir; istemci koparsa istek iptal edilir."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        req = Request(ids, min(body.max_tokens, settings.max_tokens_cap), body.temperature, body.top_p, body.top_k,
                      stop_ids=() if eos is None else (eos,),
                      callback=lambda tok, fin: loop.call_soon_threadsafe(q.put_nowait, (tok, fin)))
        stops = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
        detok, matcher = Detokenizer(tokenizer), StopMatcher(stops)
        engine.submit(req)
        try:
            while True:
                tok, fin = await q.get()
                if tok is not None:
                    text, hit = matcher.feed(detok.push(tok))
                    if hit:
                        req.cancelled = True
                        yield text, "stop", req
                        return
                    if text:
                        yield text, None, req
                if fin is not None:
                    yield matcher.flush(), fin, req
                    return
        finally:
            req.cancelled = True

    def encode(prompt: str) -> List[int]:
        ids = tokenizer.encode(prompt).ids
        if len(ids) >= engine.max_context:
            raise HTTPException(400, f"istem çok uzun ({len(ids)} token, sınır {engine.max_context - 1})")
        return ids

    def chat_prompt(messages: List[ChatMessage]) -> str:
        system = "".join(f"<|sistem|>\n{m.content}\n" for m in messages if m.role == "system")
        history = "".join(f"<|{'kullanici' if m.role == 'user' else 'asistan'}|>\n{m.content}\n"
                          for m in messages if m.role != "system")
        return settings.chat_template.format(system=system, history=history)

    async def respond(body: CompletionBody, prompt: str, kid: str, chat: bool):
        if body.n != 1:
            raise HTTPException(400, "yalnız n=1 destekleniyor")
        ids = encode(prompt)
        limiter.acquire(kid)
        t0, cid = time.time(), ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        endpoint = "chat" if chat else "completions"
        if chat:
            body.stop = ([body.stop] if isinstance(body.stop, str) else list(body.stop or [])) + ["<|kullanici|>"]

        def chunk(text=None, fin=None, first=False):
            if chat:
                delta = {"role": "assistant", "content": ""} if first else ({"content": text} if text else {})
                choice = {"index": 0, "delta": delta, "finish_reason": fin}
                return {"id": cid, "object": "chat.completion.chunk", "created": int(t0), "model": settings.model_name,
                        "choices": [choice]}
            return {"id": cid, "object": "text_completion", "created": int(t0), "model": settings.model_name,
                    "choices": [{"index": 0, "text": text or "", "finish_reason": fin, "logprobs": None}]}

        if body.stream:
            async def sse():
                n_out, fin, req = 0, "cancelled", None
                try:
                    if chat:
                        yield f"data: {json.dumps(chunk(first=True))}\n\n"
                    async for text, f, req in generate(ids, body):
                        if text:
                            yield f"data: {json.dumps(chunk(text))}\n\n"
                        if f is not None:
                            fin = f
                            yield f"data: {json.dumps(chunk(fin=f if f != 'error' else 'stop'))}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    limiter.release(kid)
                    record(kid, endpoint, len(ids), len(req.out) if req else n_out, fin, t0,
                           req.t_first if req else None)
            return StreamingResponse(sse(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        parts, fin, req = [], "stop", None
        try:
            async for text, f, req in generate(ids, body):
                parts.append(text)
                if f is not None:
                    fin = f
        finally:
            limiter.release(kid)
            record(kid, endpoint, len(ids), len(req.out) if req else 0, fin, t0, req.t_first if req else None)
        if fin == "error":
            raise HTTPException(500, "üretim sırasında sunucu hatası")
        text, n_out = "".join(parts), len(req.out) if req else 0
        usage = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out}
        if chat:
            return {"id": cid, "object": "chat.completion", "created": int(t0), "model": settings.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fin}],
                    "usage": usage}
        return {"id": cid, "object": "text_completion", "created": int(t0), "model": settings.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": fin, "logprobs": None}], "usage": usage}

    @app.post("/v1/completions")
    async def completions(body: CompletionBody, kid: str = Depends(auth)):
        prompt = body.prompt if isinstance(body.prompt, str) else "".join(body.prompt[:1])
        return await respond(body, prompt, kid, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatBody, kid: str = Depends(auth)):
        return await respond(body, chat_prompt(body.messages), kid, chat=True)

    @app.get("/v1/models")
    async def models(kid: str = Depends(auth)):
        return {"object": "list", "data": [{"id": settings.model_name, "object": "model", "owned_by": "valerois"}]}

    @app.get("/health")
    async def health():
        alive = engine._thread.is_alive()
        body = {"status": "ok" if alive else "down", "model": settings.model_name,
                "active": len(engine.active), "queued": engine.queued}
        return JSONResponse(body, status_code=200 if alive else 503)

    @app.get("/metrics")
    async def prom():
        return PlainTextResponse(metrics.render(engine))

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Heimdall OpenAI uyumlu sunucu")
    p.add_argument("--model", default=os.environ.get("HEIMDALL_MODEL"), help="servis paketi klasörü ya da last.pt")
    p.add_argument("--tokenizer")
    p.add_argument("--name", default=os.environ.get("HEIMDALL_MODEL_NAME", "heimdall"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    p.add_argument("--device", default=None)
    p.add_argument("--packed", action="store_true", help="bit modelinde ağırlıkları 1/2 bit paketle (bellek 8-16× az)")
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--max-context", type=int, default=16384)
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--max-tokens-cap", type=int, default=2048)
    p.add_argument("--rpm", type=int, default=int(os.environ.get("HEIMDALL_RPM", 120)))
    p.add_argument("--max-concurrent", type=int, default=int(os.environ.get("HEIMDALL_MAX_CONCURRENT", 8)))
    p.add_argument("--keys-file", default=os.environ.get("HEIMDALL_KEYS_FILE"))
    p.add_argument("--usage-log", default=os.environ.get("HEIMDALL_USAGE_LOG"))
    args = p.parse_args()
    if not args.model:
        raise SystemExit("--model ya da HEIMDALL_MODEL gerekli")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn
    from tokenizers import Tokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = load_model(args.model, device, dtype)
    log.info("delta çekirdeği: %s", kernels.prepare(device, model.cfg.kernel))
    if args.packed:
        from .bitkernel import pack_bit_model
        log.info("paketleme: %.1f MB bellek kazanıldı", pack_bit_model(model) / 2**20)
    tok = Tokenizer.from_file(tokenizer_path(args.model, args.tokenizer))
    engine = Engine(model, args.max_batch, args.max_context, args.prefill_chunk).start()
    settings = Settings(model_name=args.name, api_key_hashes=load_keys(os.environ.get("HEIMDALL_API_KEYS"),
                                                                       args.keys_file),
                        rpm=args.rpm, max_concurrent=args.max_concurrent, max_tokens_cap=args.max_tokens_cap,
                        usage_log=args.usage_log)
    log.info("model %s: %.1fM param, düzen %s, %s %s", args.name, model.num_params() / 1e6, model.cfg.layout,
             device, dtype)
    uvicorn.run(create_app(engine, tok, settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

"""Model backends for the naming game.

The local MLX backend stays the default and is untouched, so every run recorded
so far remains reproducible. This module adds a hosted backend behind the same
tiny interface -- `choose(agent, pool, rng) -> (choice, parse_failure)` -- plus a
batched variant the run loop uses when the backend supports concurrency.

Two properties matter and neither is free:

Determinism.  The local path draws prompt-shuffle and fallback randomness from
one shared seeded RNG, in call order. Firing a round's calls concurrently would
make that order depend on thread scheduling. So the hosted path derives a
*per-agent* RNG from (seed, round, agent_id) instead: the same agent sees the
same prompt shuffle no matter when its call returns.

Cost.  Hosted calls spend real money. The backend counts every call, enforces a
hard cap, and a dry run prints what it would send without sending anything.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

#: Hosted backends that speak the OpenAI chat-completions shape. Adding one is
#: an entry here, not a new class.
BACKENDS = {
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "headers": {"X-Title": "bluedot-naming-game"},
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "headers": {},
    },
}

API_KEY_ENV = "OPENROUTER_API_KEY"  # default, overridden per backend


class BudgetExceeded(RuntimeError):
    """The configured call cap was reached. Raised before spending more."""


class ProviderUnusable(RuntimeError):
    """The backend cannot serve this config at all.

    Raised loudly rather than degraded, because the fallback for a failed call
    is a random choice. A wrong model id would otherwise fill a whole log with
    coin flips, marked only as parse failures, and look like a real run.
    """


class MissingApiKey(RuntimeError):
    def __init__(self, key_env: str = API_KEY_ENV) -> None:
        super().__init__(
            f"this provider needs ${key_env} in the environment. "
            "Note that a non-interactive shell does not source ~/.zshrc."
        )


@dataclass
class HostedPlayer:
    """One hosted model per agent, so a population can be heterogeneous.

    `models` maps agent_id -> model id. Agents absent from the map fall back to
    `default_model`.
    """

    default_model: str
    backend: str = "openrouter"
    models: dict[int, str] = field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: int = 60
    max_concurrency: int = 8
    timeout: float = 60.0
    max_calls: int | None = None
    dry_run: bool = False
    seed: int = 0

    calls: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    consecutive_failures: int = field(default=0, init=False)
    _round: int = field(default=0, init=False)

    #: Abort once this many calls fail back to back. A handful of transient
    #: errors is normal; a wall of them means the run is producing noise.
    failure_abort_threshold: int = 12

    #: Attempts per call. Rate limits are retried with exponential backoff; a
    #: 4xx that will not fix itself breaks out immediately.
    max_retries: int = 6

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {sorted(BACKENDS)}")
        spec = BACKENDS[self.backend]
        self.endpoint = spec["endpoint"]
        self.key_env = spec["key_env"]
        self.extra_headers = dict(spec["headers"])
        self.api_key = os.environ.get(self.key_env, "")
        if not self.api_key and not self.dry_run:
            raise MissingApiKey(self.key_env)

    def model_for(self, agent_id: int) -> str:
        return self.models.get(agent_id, self.default_model)

    def set_round(self, rnd: int) -> None:
        """The run loop announces the round so per-agent RNGs stay deterministic."""
        self._round = rnd

    def agent_rng(self, agent_id: int) -> random.Random:
        return random.Random((self.seed, self._round, agent_id).__hash__())

    # -- transport ---------------------------------------------------------

    def _post(self, model: str, messages: list[dict], pool: Sequence[str] = ()) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"reached max_calls={self.max_calls}; raise it deliberately or shorten the run"
            )
        self.calls += 1
        if self.dry_run:
            # Return a *valid* answer so the loop takes the same path it would
            # in a real run. A reply that failed to parse would trigger the
            # retry, and the call count -- the whole point of a dry run -- would
            # come out at twice the real cost.
            name = list(pool)[0] if pool else "F"
            return f"DRY RUN: no request sent.\nANSWER: {name}"

        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429:
                    # Rate limiting is transient and expected when a sweep runs
                    # several games at once. Back off properly and honour
                    # Retry-After rather than burning the attempt budget.
                    hinted = exc.headers.get("Retry-After") if exc.headers else None
                    delay = float(hinted) if hinted and str(hinted).isdigit() else 2.0 * 2**attempt
                    time.sleep(min(delay, 60.0))
                    continue
                if 400 <= exc.code < 500 and exc.code != 408:
                    break  # a bad key or model id will not fix itself
                time.sleep(1.5 * (attempt + 1))
            except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        self.failures += 1
        raise RuntimeError(
            f"{self.backend} call failed after {self.max_retries} attempts: {last_error}"
        )

    # -- the player interface ---------------------------------------------

    def preflight(self, pool: Sequence[str]) -> None:
        """Make one real call before the run starts.

        Catches a wrong model id, a bad key or a dead endpoint while nothing has
        been spent and no log exists, instead of after 1400 random choices.
        """
        if self.dry_run:
            return
        probe = [{"role": "user", "content": f"Reply with exactly: ANSWER: {list(pool)[0]}"}]
        for model in sorted({self.model_for(i) for i in range(max(len(self.models), 1))} |
                            {self.default_model}):
            try:
                self._post(model, probe, pool)
            except RuntimeError as exc:
                raise ProviderUnusable(
                    f"preflight failed for model {model!r}: {exc}\n"
                    "Check the model id against OpenRouter's current catalogue and that "
                    f"${API_KEY_ENV} is valid. Nothing was run."
                ) from exc
        self.calls = 0  # preflight is not part of the experiment's budget

    def choose(self, agent, pool: Sequence[str], rng: random.Random) -> tuple[str, bool]:
        """Single decision. `rng` is ignored in favour of a per-agent stream."""
        from naming_game import RETRY_SUFFIX, build_prompt, parse_choice

        local_rng = self.agent_rng(agent.agent_id)
        model = self.model_for(agent.agent_id)
        messages = build_prompt(agent, list(pool), local_rng)

        for attempt in range(2):
            try:
                reply = self._post(model, messages, pool)
            except BudgetExceeded:
                raise
            except RuntimeError as exc:
                # A transport failure is not a model refusal. Fall back and mark
                # it, but stop the run if they are not isolated.
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_abort_threshold:
                    raise ProviderUnusable(
                        f"{self.consecutive_failures} consecutive call failures; "
                        f"last error: {exc}. Aborting rather than logging coin flips."
                    ) from exc
                return local_rng.choice(list(pool)), True
            choice = parse_choice(reply, list(pool))
            if choice is not None:
                self.consecutive_failures = 0
                return choice, False
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": reply.strip()[:200]},
                    {"role": "user", "content": RETRY_SUFFIX.format(pool=", ".join(pool))},
                ]
        return local_rng.choice(list(pool)), True

    def choose_many(
        self, agents: Sequence, pool: Sequence[str], rng: random.Random
    ) -> dict[int, tuple[str, bool]]:
        """A whole round at once. Results are keyed by agent id, so the order in
        which calls return cannot change the recorded outcome."""
        if not agents:
            return {}
        workers = max(1, min(self.max_concurrency, len(agents)))
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            futures = {
                agent.agent_id: pool_exec.submit(self.choose, agent, pool, rng)
                for agent in agents
            }
            return {agent_id: future.result() for agent_id, future in futures.items()}


def assign_models(model_ids: Sequence[str], population: int) -> dict[int, str]:
    """Deal models round-robin across the population.

    Deterministic and balanced: agent i gets `model_ids[i % len(model_ids)]`, so
    a heterogeneous population is reproducible and the factions are not
    accidentally confounded with a model family.
    """
    if not model_ids:
        return {}
    return {i: model_ids[i % len(model_ids)] for i in range(population)}


def build_player(cfg: dict, seed: int, dry_run: bool = False):
    """Construct the backend a config asks for. Defaults to local MLX."""
    provider = str(cfg.get("provider", "mlx")).lower()
    if provider == "mlx":
        from naming_game import ModelPlayer

        return ModelPlayer(cfg["model"], cfg["temperature"], cfg["max_tokens"])
    if provider not in BACKENDS:
        raise ValueError(
            f"unknown provider {provider!r}; expected 'mlx' or one of {sorted(BACKENDS)}"
        )

    options = cfg.get("hosted") or cfg.get(provider) or {}
    model_ids = [str(m) for m in (cfg.get("models") or [])]
    return HostedPlayer(
        default_model=str(cfg["model"]),
        backend=provider,
        models=assign_models(model_ids, int(cfg["population_size"])),
        temperature=float(cfg["temperature"]),
        max_tokens=int(cfg["max_tokens"]),
        max_concurrency=int(options.get("max_concurrency", 8)),
        timeout=float(options.get("timeout", 60.0)),
        max_calls=options.get("max_calls"),
        dry_run=dry_run,
        seed=seed,
    )


#: Kept so existing imports and configs keep working.
OpenRouterPlayer = HostedPlayer

from openai import OpenAI

from ..capabilities import check_job_compatibility
from ..config import RunnerConfig
from .base import RunnerResult, TranslationJob
from .structured import structured_params


def _reported_model(value: str | None) -> dict:
    if not value:
        return {"status": "unknown", "reason": "provider omitted model identity"}
    if "/" in value:
        provider, model = value.split("/", 1)
        if provider and model:
            return {"provider": provider, "model": model, "source": "provider-reported"}
    return {"id": value, "source": "provider-reported"}


class DirectOpenAIRunner:
    """Chat Completions adapter; retries belong exclusively to orchestration."""

    def __init__(self, config: RunnerConfig, api_key: str, client: OpenAI | None = None):
        self.config = config
        self.api_key = api_key
        self.client = client

    def run(self, job: TranslationJob) -> RunnerResult:
        if not self.config.model:
            raise ValueError("Set a model ID for this OpenAI-compatible profile")
        check_job_compatibility(self.config, job)
        options = structured_params(self.config, job)
        owned = self.client is None
        client = self.client or OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )
        params = {"model": self.config.model, "messages": job.messages()}
        temperature = (
            self.config.temperature
            if "temperature" in self.config.model_fields_set
            else job.temperature
        )
        if temperature is not None:
            params["temperature"] = temperature
        # Reasoning APIs may require max_completion_tokens and no temperature.
        if "max_completion_tokens" not in self.config.extra_params:
            params["max_tokens"] = self.config.max_tokens or job.max_tokens
        try:
            response = client.chat.completions.create(**params, extra_body=options)
        finally:
            if owned:
                client.close()
        effective_settings = {
            "model": self.config.model,
            "model_source": "configured",
            "reported_model": response.model,
            "execution": {"resolved_model": _reported_model(response.model)},
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "extra_params": options,
        }
        if not response.choices:
            return RunnerResult(
                text=None,
                model=response.model,
                raw_response=response.model_dump(mode="json"),
                effective_settings=effective_settings,
            )
        choice = response.choices[0]
        usage = response.usage
        counters = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        for key, value in {
            "reasoning_tokens": getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None
            ),
            "cached_input_tokens": getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
            "cache_write_input_tokens": getattr(
                getattr(usage, "prompt_tokens_details", None), "cache_write_tokens", None
            ),
        }.items():
            if value is not None:
                counters[key] = value
        return RunnerResult(
            text=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=counters,
            effective_settings=effective_settings,
            raw_response=response.model_dump(mode="json"),
        )

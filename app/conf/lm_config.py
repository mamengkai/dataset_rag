from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    llm_model: str
    vl_model: str
    llm_temperature: float


lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    vl_model=os.getenv("VL_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
)

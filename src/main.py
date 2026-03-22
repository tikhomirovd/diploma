from __future__ import annotations

import os

from beartype import beartype
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@beartype
def build_llm() -> ChatOpenAI:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model = os.getenv("MODEL", "openai/gpt-4o")
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,  # type: ignore[arg-type]
        openai_api_base="https://openrouter.ai/api/v1",
    )


@beartype
def main() -> None:
    llm = build_llm()
    response = llm.invoke("Say hello in one sentence.")
    print(response.content)


if __name__ == "__main__":
    main()

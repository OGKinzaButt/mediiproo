"""
chains.py
---------
Builds the ChatOpenAI model, a reusable LLMChain for the JSON assessment,
a small demo of raw System/Human/AIMessage usage, and a generator that
streams the narrative explanation chunk by chunk.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LLMChain moved around between LangChain versions. Try the classic
# location first (langchain.chains), and fall back to langchain_classic
# if the environment only has newer split packages installed.
try:
    from langchain.chains import LLMChain
except ModuleNotFoundError:
    from langchain_classic.chains import LLMChain

from src.prompts import (
    ASSESSMENT_CHAT_TEMPLATE,
    NARRATIVE_CHAT_TEMPLATE,
    SYSTEM_SAFETY_RULES,
)


def build_llm(api_key: str, model_name: str = "gpt-4o-mini",
              temperature: float = 0.3, streaming: bool = False) -> ChatOpenAI:
    """Create a ChatOpenAI instance bound to the user's own API key."""
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


def build_assessment_chain(llm: ChatOpenAI) -> LLMChain:
    """The core, reusable LLMChain that produces the structured JSON."""
    return LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_TEMPLATE)


def run_message_demo(llm: ChatOpenAI, user_text: str) -> AIMessage:
    """
    Small standalone demo of building a conversation by hand with
    SystemMessage / HumanMessage, and receiving an AIMessage back.
    Not part of the main flow -- included to satisfy the assignment's
    "raw message objects" requirement and useful for debugging.
    """
    messages = [
        SystemMessage(content=SYSTEM_SAFETY_RULES),
        HumanMessage(content=user_text),
    ]
    response: AIMessage = llm.invoke(messages)
    return response


def stream_narrative(llm: ChatOpenAI, inputs: dict):
    """
    Generator that yields narrative text chunks as they arrive from the
    model. Pass this straight into st.write_stream() in app.py.
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content

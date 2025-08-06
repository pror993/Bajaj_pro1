import os
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from .prompt_templates import phase_a, phase_b

cfg = yaml.safe_load(open("interpretation_config.yaml"))

# Use Gemini API from LangChain Google GenAI, but you can swap for any open-source LLM

def get_llm_chain(prompt_template):
    llm = ChatGoogleGenerativeAI(
        model=cfg["llm"]["model_name"],
        temperature=cfg["llm"]["temperature"],
        max_output_tokens=cfg["llm"]["max_tokens"],
        google_api_key=os.getenv("GEMINI_API_KEY", "")
    )
    return LLMChain(
        llm=llm,
        prompt=prompt_template
    )

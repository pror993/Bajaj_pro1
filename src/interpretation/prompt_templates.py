import yaml
from langchain.prompts import PromptTemplate
import os

# Always use absolute path for config
config_path = os.path.join(os.path.dirname(__file__), '../../interpretation_config.yaml')
cfg = yaml.safe_load(open(config_path, encoding='utf-8'))

phase_a_path = os.path.join(os.path.dirname(__file__), '../../' + cfg["templates"]["phase_a"])
phase_b_path = os.path.join(os.path.dirname(__file__), '../../' + cfg["templates"]["phase_b"])

with open(phase_a_path, encoding='utf-8') as f:
    phase_a_template = f.read()
phase_a = PromptTemplate(
    template=phase_a_template,
    input_variables=["clauses", "domain"]
)

with open(phase_b_path, encoding='utf-8') as f:
    phase_b_template = f.read()
phase_b = PromptTemplate(
    template=phase_b_template,
    input_variables=["summaries", "slots", "domain"]
)

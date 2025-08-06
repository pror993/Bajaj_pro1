from .prompt_templates import phase_a, phase_b
from .llm_client import get_llm_chain
import json

class Interpreter:
    def __init__(self):
        self.chain_a = get_llm_chain(phase_a)
        self.chain_b = get_llm_chain(phase_b)

    def interpret_clauses(self, clauses, domain):
        """
        Phase A: Given retrieved clauses,
        return a list of { clause_id, summary }.
        """
        response = self.chain_a.run({"clauses": clauses, "domain": domain})
        try:
            return json.loads(response)
        except Exception:
            return []

    def make_decision(self, summaries, slots, domain):
        """
        Phase B: Given summaries and slots,
        generate reasoning_trace + decision + amount.
        """
        response = self.chain_b.run({
            "summaries": summaries,
            "slots": slots,
            "domain": domain
        })
        try:
            return json.loads(response)
        except Exception:
            return {}

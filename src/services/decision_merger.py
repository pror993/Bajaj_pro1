from src.rules.engine import RuleEngine

class DecisionMerger:
    def __init__(self):
        self.rule_engine = RuleEngine()

    def merge(self, domain: str, llm_block: dict, slot_facts: dict):
        triggered = self.rule_engine.evaluate(domain, slot_facts)
        if "exclude_preexisting" in triggered:
            final_decision = "rejected"
            final_amount = 0
            note = "Pre-existing condition exclusion rule applied"
        elif llm_block["decision"] == "approved" and "allow_waiting" not in triggered:
            final_decision = "rejected"
            final_amount = 0
            note = "Waiting period rule not satisfied"
        else:
            final_decision = llm_block["decision"]
            final_amount = llm_block["amount"]
            note = "LLM decision confirmed by rules"
        merged = {
            **llm_block,
            "rule_events": triggered,
            "final_decision": final_decision,
            "final_amount": final_amount,
            "rule_note": note
        }
        return merged

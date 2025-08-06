import unittest
from src.rules.rule_loader import load_rules
from src.rules.engine import RuleEngine
from src.services.decision_merger import DecisionMerger

class TestRuleEngine(unittest.TestCase):
    def test_rule_loader(self):
        rules = load_rules()
        self.assertIn("health", rules)
        self.assertTrue(isinstance(rules["health"], list))

    def test_rule_engine_health(self):
        engine = RuleEngine()
        events = engine.evaluate("health", {"policy_age_days": 30})
        self.assertIn("exclude_preexisting", events)

    def test_decision_merger(self):
        merger = DecisionMerger()
        llm_block = {"reasoning_trace": ["..."], "decision": "approved", "amount": 1000}
        slot_facts = {"policy_age_days": 30, "claim_amount": 1000}
        merged = merger.merge("health", llm_block, slot_facts)
        self.assertEqual(merged["final_decision"], "rejected")
        self.assertEqual(merged["final_amount"], 0)
        self.assertIn("rule_note", merged)

if __name__ == "__main__":
    unittest.main()

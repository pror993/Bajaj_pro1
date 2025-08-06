from .rule_loader import load_rules
from business_rules.engine import run_all
from business_rules.variables import BaseVariables, numeric_rule_variable
from business_rules.actions import BaseActions, rule_action

class FactVariables(BaseVariables):
    def __init__(self, facts):
        self.facts = facts
    @numeric_rule_variable
    def policy_age_days(self):
        return self.facts.get('policy_age_days', 0)
    @numeric_rule_variable
    def claim_amount(self):
        return self.facts.get('claim_amount', 0)
    @numeric_rule_variable
    def deductible(self):
        return self.facts.get('deductible', 0)
    @numeric_rule_variable
    def days_before_trip(self):
        return self.facts.get('days_before_trip', 0)

class EventActions(BaseActions):
    def __init__(self):
        self.events = []
    @rule_action(params=None)
    def allow_waiting(self):
        self.events.append('allow_waiting')
    @rule_action(params=None)
    def exclude_preexisting(self):
        self.events.append('exclude_preexisting')
    @rule_action(params=None)
    def sufficient_amount(self):
        self.events.append('sufficient_amount')
    @rule_action(params=None)
    def full_refund(self):
        self.events.append('full_refund')

class RuleEngine:
    def __init__(self):
        self.raw_rules = load_rules()
    def evaluate(self, domain: str, facts: dict):
        rules = self.raw_rules.get(domain, [])
        actions = EventActions()
        run_all(
            rule_list=rules,
            defined_variables=FactVariables(facts),
            defined_actions=actions,
            stop_on_first_trigger=False
        )
        return actions.events

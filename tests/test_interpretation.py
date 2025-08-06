import unittest
from unittest.mock import patch, MagicMock
from src.interpretation.interpreter import Interpreter

class TestInterpreter(unittest.TestCase):
    def setUp(self):
        # Patch get_llm_chain to return a mock chain with a dummy run method
        patcher = patch('src.interpretation.llm_client.get_llm_chain')
        self.mock_get_llm_chain = patcher.start()
        self.addCleanup(patcher.stop)
        mock_chain = MagicMock()
        mock_chain.run.side_effect = [
            '[{"clause_id": "1", "summary": "Agreement to provide services."}]',
            '{"reasoning_trace": ["..."], "decision": "approved", "amount": 1000}'
        ]
        self.mock_get_llm_chain.return_value = mock_chain
        self.interpreter = Interpreter()

    def test_phase_a_template(self):
        clauses = '[{"clause_id": "1", "text": "WHEREAS the parties agree..."}]'
        domain = "health"
        summaries = self.interpreter.interpret_clauses(clauses, domain)
        self.assertIsInstance(summaries, list)
        self.assertTrue(any('clause_id' in s for s in summaries))

    def test_phase_b_template(self):
        summaries = '[{"clause_id": "1", "summary": "Agreement to provide services."}]'
        slots = '{"age": 35, "procedure": "knee surgery"}'
        domain = "health"
        decision_block = self.interpreter.make_decision(summaries, slots, domain)
        self.assertIsInstance(decision_block, dict)
        self.assertIn("decision", decision_block)
        self.assertIn("reasoning_trace", decision_block)

if __name__ == "__main__":
    unittest.main()

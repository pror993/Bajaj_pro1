import yaml
import os

def load_rules(config_path="rule_engine_config.yaml"):
    config_path = os.path.abspath(config_path)
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg.get("rules", [])
    except Exception as e:
        print(f"Error loading rules config: {e}")
        return []

import yaml

def load_rules(config_path="rule_engine_config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    return cfg["rules"]

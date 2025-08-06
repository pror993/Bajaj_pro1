import yaml
from typing import Dict, Any, Optional


class PromptBuilder:
    def __init__(self, config_path: str = "retrieval_config.yaml"):
        """Initialize prompt builder with configuration."""
        self.config = self._load_config(config_path)
        self.templates = self.config.get("prompt_templates", {})
    
    def _load_config(self, config_path: str) -> Dict:
        """Load retrieval configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading retrieval config: {e}")
            return {}
    
    def build_search_prompt(self, raw_query: str, slots: Dict[str, Any], domain: str) -> str:
        """
        Enrich the user query with slot values and domain context.
        
        Args:
            raw_query: Original user query
            slots: Extracted slot values
            domain: Detected domain (health, motor, travel)
            
        Returns:
            Enriched search prompt
        """
        if domain not in self.templates:
            # Fallback to generic template
            return self._build_generic_prompt(raw_query, slots, domain)
        
        template_config = self.templates[domain]
        template = template_config.get("template", "")
        fallback = template_config.get("fallback", "")
        
        try:
            # Try to use the main template
            enriched_prompt = template.format(**slots)
        except KeyError as e:
            # If template has missing slots, use fallback
            print(f"Missing slot {e} for template, using fallback")
            try:
                enriched_prompt = fallback.format(**slots)
            except KeyError:
                # If fallback also fails, use generic approach
                enriched_prompt = self._build_generic_prompt(raw_query, slots, domain)
        
        # Combine with original query
        final_prompt = f"{enriched_prompt} {raw_query}"
        return final_prompt.strip()
    
    def _build_generic_prompt(self, raw_query: str, slots: Dict[str, Any], domain: str) -> str:
        """Build a generic prompt when domain-specific template fails."""
        slot_terms = []
        
        for slot_name, slot_value in slots.items():
            if slot_value:
                if isinstance(slot_value, dict):
                    # Handle complex slot values
                    if 'value' in slot_value:
                        slot_terms.append(f"{slot_name}: {slot_value['value']}")
                    elif 'text' in slot_value:
                        slot_terms.append(f"{slot_name}: {slot_value['text']}")
                else:
                    slot_terms.append(f"{slot_name}: {slot_value}")
        
        if slot_terms:
            slot_context = f"{domain} claim with " + ", ".join(slot_terms)
            return f"{slot_context}. {raw_query}"
        else:
            return f"{domain} insurance claim: {raw_query}"
    
    def build_domain_specific_prompt(self, domain: str, **kwargs) -> str:
        """
        Build a domain-specific prompt with given parameters.
        
        Args:
            domain: Target domain
            **kwargs: Template parameters
            
        Returns:
            Formatted prompt
        """
        if domain not in self.templates:
            return f"{domain} insurance claim"
        
        template_config = self.templates[domain]
        template = template_config.get("template", "")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Missing parameter {e} for {domain} template")
            return f"{domain} insurance claim"
    
    def get_available_domains(self) -> list:
        """Get list of available domains with templates."""
        return list(self.templates.keys())
    
    def get_template_info(self, domain: str) -> Dict[str, Any]:
        """Get template information for a specific domain."""
        if domain not in self.templates:
            return {}
        
        template_config = self.templates[domain]
        
        # Extract required parameters from template
        template = template_config.get("template", "")
        required_params = self._extract_template_params(template)
        
        return {
            "template": template,
            "fallback": template_config.get("fallback", ""),
            "required_parameters": required_params
        }
    
    def _extract_template_params(self, template: str) -> list:
        """Extract parameter names from a template string."""
        import re
        params = re.findall(r'\{(\w+)\}', template)
        return list(set(params))  # Remove duplicates
    
    def validate_slots_for_domain(self, slots: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Validate that required slots are present for a domain.
        
        Args:
            slots: Available slot values
            domain: Target domain
            
        Returns:
            Dict with validation results
        """
        if domain not in self.templates:
            return {"valid": False, "missing": [], "message": f"Domain {domain} not supported"}
        
        template_info = self.get_template_info(domain)
        required_params = template_info.get("required_parameters", [])
        
        missing_params = []
        for param in required_params:
            if param not in slots or not slots[param]:
                missing_params.append(param)
        
        return {
            "valid": len(missing_params) == 0,
            "missing": missing_params,
            "message": f"Missing required parameters: {missing_params}" if missing_params else "All required parameters present"
        } 
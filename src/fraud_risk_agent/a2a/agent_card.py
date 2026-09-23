"""A2A agent card served at /.well-known/agent.json."""

AGENT_CARD: dict = {
    "name": "fraud-risk-agent",
    "description": (
        "E-commerce fraud risk assessment agent. Analyzes return, cancellation, "
        "and identity-linking patterns to produce advisory risk assessments."
    ),
    "version": "1.0.0",
    "url": "",  # Set at runtime from config
    "skills": [
        {
            "name": "assess_customer_risk",
            "description": (
                "Assess fraud risk for a customer based on return, "
                "cancellation, and identity signals"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer identifier",
                    },
                    "order_id": {
                        "type": "string",
                        "description": "Optional order to focus on",
                    },
                    "context": {
                        "type": "string",
                        "enum": [
                            "refund_request",
                            "return_request",
                            "support_contact",
                        ],
                        "description": "What triggered this assessment",
                    },
                },
                "required": ["customer_id"],
            },
            "output_schema": {
                "type": "object",
                "description": "FraudAssessment with score, signals, evidence, and recommendation",
            },
        }
    ],
    "auth": {
        "type": "bearer",
        "description": "OAuth2 client-credentials bearer token",
    },
}


def get_agent_card(base_url: str) -> dict:
    """Return the agent card with the base URL filled in."""
    card = AGENT_CARD.copy()
    card["url"] = base_url
    return card

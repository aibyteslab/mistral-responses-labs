"""Fake bookshop data and the tool functions the agent can call.

Everything here is invented for the video. No real customers, no real orders.
"""

ORDERS = {
    "101": {"order_id": "101", "item": "The Pragmatic Programmer", "status": "delivered",
            "delivered_days_ago": 40},
    "104": {"order_id": "104", "item": "Designing Data-Intensive Applications",
            "status": "delivered", "delivered_days_ago": 9},
    "107": {"order_id": "107", "item": "Site Reliability Engineering", "status": "in transit",
            "delivered_days_ago": None},
}

REFUND_POLICY = {
    "return_window_days": 30,
    "condition": "unused, original packaging",
    "refund_method": "original payment method within 5 business days",
}


def lookup_order(order_id: str) -> dict:
    """Return the order record, or an error dict when the number is unknown."""
    order = ORDERS.get(str(order_id).strip())
    if order is None:
        return {"error": f"order {order_id} not found"}
    return order


def get_refund_policy() -> dict:
    """Return the shop's refund policy."""
    return REFUND_POLICY


def create_ticket(order_id: str, status: str, refund_allowed: bool, summary: str) -> dict:
    """Pretend to file a ticket; in the video this just echoes the structured record."""
    return {"ticket_id": f"T-{order_id}-0001", "order_id": order_id, "status": status,
            "refund_allowed": refund_allowed, "summary": summary}


# Tool definitions in the Responses API shape: type, name, description, parameters.
# `strict` is accepted by SDK 3.6.0 but not described on the docs page (2026-10-29).
TOOLS = [
    {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up a bookshop order by its order number.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order number, e.g. 104"}},
            "required": ["order_id"],
        },
    },
    {
        "type": "function",
        "name": "get_refund_policy",
        "description": "Return the shop's refund and return policy.",
        "parameters": {"type": "object", "properties": {}},
    },
]

TICKET_TOOL = {
    "type": "function",
    "name": "create_ticket",
    "description": "File a support ticket once the order status and refund eligibility are known.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "status": {"type": "string", "enum": ["delivered", "in transit", "unknown"]},
            "refund_allowed": {"type": "boolean"},
            "summary": {"type": "string", "description": "One sentence for the support team."},
        },
        "required": ["order_id", "status", "refund_allowed", "summary"],
        "additionalProperties": False,
    },
}

FUNCS = {
    "lookup_order": lookup_order,
    "get_refund_policy": get_refund_policy,
    "create_ticket": create_ticket,
}

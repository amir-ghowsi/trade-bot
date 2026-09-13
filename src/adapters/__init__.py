"""
Adapters Package Initialization.
Contains broker protocol interfaces and broker-specific adapters.
"""

from src.adapters.broker_protocol import IBrokerAdapter
from src.adapters.simulated_broker import SimulatedBroker
from src.adapters.types import BrokerDeal, BrokerOrder, BrokerPosition

__all__ = [
    "IBrokerAdapter",
    "SimulatedBroker",
    "BrokerDeal",
    "BrokerOrder",
    "BrokerPosition",
]

"""
VulnGuard Observer Agent Package.

ObserverAgent: Strategic oversight agent that reviews VulnKB state
periodically rather than claiming Intents.
"""

from .agent import ObserverAgent, ObserverAction

__all__ = ["ObserverAgent", "ObserverAction"]
"""
VulnGuard — Business Logic Vulnerability Mining & Verification Agent Framework

Core design:
- VulnKB: Fact/Intent/Hint shared knowledge graph with verified admission
- Miner Agents: specialized OODA loops for different attack surfaces
- Observer Agent: strategy evaluation + course correction with cooldown
- Verifier Agent: independent PoC construction and validation
- Task Queue: decentralized claim-based coordination
- Three-layer prompt caching (stable/context/volatile)
- Descriptor-driven tool system with risk levels and audit phase filtering
"""

__version__ = "0.1.0"

from .agent_base import AgentBase, AgentType, AgentConfig, OODAResult
from .utils import LLMGateway, LLMProvider, AgentRole, PromptManager, PromptLayer
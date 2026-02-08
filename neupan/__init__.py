"""
EndoFlow — End-to-end robot motion planner (based on NeuPAN).
"""

from neupan.planner import Planner

# Backward compatibility: `from neupan import neupan; neupan.init_from_yaml(...)`
neupan = Planner

__all__ = ["Planner", "neupan"]

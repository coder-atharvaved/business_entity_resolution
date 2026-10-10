"""
Business Entity Resolution System
==================================
Production-grade Entity Resolution pipeline for large-scale multi-source record linkage.
Complies with Amazon ML Challenge 2026 specifications:
- Two-stage decoupled architecture (Scalable Blocking + XGBoost Classifier)
- Macro-averaged F_0.5 metric optimization (precision weighted 2x over recall)
- Compact candidate generation for leaderboard bonus
- Zero external data lookups, open country compatibility (US, India, France)
"""

__version__ = "1.0.0"
__author__ = "Atharva & Antigravity ML Team"

#!/usr/bin/env python3
"""
NecroTrace - Report module re-export wrapper.
Provides backward and forward compatibility for:
    from src.report import generate_forensic_pdf, compute_sha256_hash, generate_forensic_timeline_chart
and:
    from src.reporting.pdf_generator import generate_forensic_pdf, compute_sha256_hash, generate_forensic_timeline_chart
"""

from src.reporting.pdf_generator import (
    generate_forensic_pdf,
    compute_sha256_hash,
    generate_forensic_timeline_chart,
)

__all__ = [
    "generate_forensic_pdf",
    "compute_sha256_hash",
    "generate_forensic_timeline_chart",
]

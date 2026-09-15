#!/usr/bin/env python3
"""
NecroTrace - Report module re-export wrapper.
Provides backward and forward compatibility for:
    from src.report import generate_forensic_pdf, compute_sha256_hash, generate_forensic_timeline_chart
and:
    from src.reporting.pdf_generator import generate_forensic_pdf, compute_sha256_hash, generate_forensic_timeline_chart
"""

import sys
import importlib

try:
    from src.reporting.pdf_generator import (
        generate_forensic_pdf,
        compute_sha256_hash,
        generate_forensic_timeline_chart,
        generate_qr_code_drawing,
        generate_qr_code_svg,
    )
except (ImportError, AttributeError):
    sys.modules.pop("src.reporting.pdf_generator", None)
    from src.reporting.pdf_generator import (
        generate_forensic_pdf,
        compute_sha256_hash,
        generate_forensic_timeline_chart,
        generate_qr_code_drawing,
        generate_qr_code_svg,
    )

__all__ = [
    "generate_forensic_pdf",
    "compute_sha256_hash",
    "generate_forensic_timeline_chart",
    "generate_qr_code_drawing",
    "generate_qr_code_svg",
]

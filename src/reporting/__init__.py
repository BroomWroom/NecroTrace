"""
Forensic Reporting Module for NecroTrace.
"""
import sys
import importlib

try:
    from .pdf_generator import (
        generate_forensic_pdf,
        compute_sha256_hash,
        generate_forensic_timeline_chart,
        generate_qr_code_drawing,
        generate_qr_code_svg,
    )
except (ImportError, AttributeError):
    sys.modules.pop("src.reporting.pdf_generator", None)
    from .pdf_generator import (
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

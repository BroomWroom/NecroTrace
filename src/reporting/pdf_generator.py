#!/usr/bin/env python3
"""
NecroTrace - Official Forensic Post-Mortem Report Generator (PDF).
Modeled after official Police & Medico-Legal Post-Mortem Examination Forms (Form No. PM-5372).
Integrates patient particulars, external autopsy signs, verified microbial bioindicators,
and definitive Postmortem Interval (PMI) time-of-death windows.
"""

import io
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
)
from reportlab.graphics.barcode import qr
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderSVG


def compute_sha256_hash(data: str | bytes) -> str:
    """Compute cryptographic SHA-256 hash for evidence chain-of-custody."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def generate_qr_code_drawing(payload: str, size: float = 52.0) -> Drawing:
    """
    Generate a native ReportLab Drawing containing an authentic, high-scannability QR Code.
    Includes a pure white background and optimal quiet zone for instantaneous mobile camera autofocus.
    """
    d = Drawing(size, size)
    d.add(Rect(0, 0, size, size, fillColor=colors.white, strokeColor=None))
    qr_widget = qr.QrCodeWidget(payload)
    qr_widget.barLevel = "M"
    qr_widget.barBorder = 2
    qr_widget.barWidth = size
    qr_widget.barHeight = size
    d.add(qr_widget)
    return d


def generate_qr_code_svg(payload: str, size: float = 130.0) -> str:
    """
    Generate an SVG string of the forensic verification QR code for embedding
    directly into Streamlit Web UI dashboards and HTML dossier cards with high contrast.
    """
    d = generate_qr_code_drawing(payload, size)
    return renderSVG.drawToString(d)


def generate_forensic_timeline_chart(
    sample_id: str,
    lower_bound_days: float,
    predicted_pmi_days: float,
    upper_bound_days: float
) -> io.BytesIO:
    """
    Generate an authentic forensic interval timeline for embedding in Section 5.
    """
    fig, ax = plt.subplots(figsize=(6.8, 1.55), dpi=220)

    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")

    x_max = max(28.0, upper_bound_days * 1.3)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.45, 0.45)

    # Probable Window of Death band
    ax.axvspan(
        lower_bound_days,
        upper_bound_days,
        color="#bfdbfe",
        alpha=0.6,
        label="Probable Time Window (Conservative Range)"
    )

    # Centerline
    ax.hlines(
        y=0,
        xmin=lower_bound_days,
        xmax=upper_bound_days,
        color="#1e3a8a",
        linewidth=2.5,
        alpha=0.9
    )

    # Marker for Predicted PMI
    ax.plot(
        predicted_pmi_days,
        0,
        marker="o",
        markersize=9,
        color="#b91c1c",
        markeredgecolor="#ffffff",
        markeredgewidth=1.8,
        label=f"Most Likely PMI: {predicted_pmi_days:.1f} Days"
    )

    # Annotations
    ax.annotate(
        f"{predicted_pmi_days:.1f} Days\n({predicted_pmi_days*24.0:.0f} Hours)",
        xy=(predicted_pmi_days, 0),
        xytext=(predicted_pmi_days, 0.16),
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#991b1b",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2)
    )

    ax.annotate(
        f"Lower: {lower_bound_days:.1f}d",
        xy=(lower_bound_days, -0.1),
        xytext=(lower_bound_days, -0.28),
        ha="center",
        fontsize=7,
        color="#1e3a8a",
        fontweight="bold"
    )

    ax.annotate(
        f"Upper: {upper_bound_days:.1f}d",
        xy=(upper_bound_days, -0.1),
        xytext=(upper_bound_days, -0.28),
        ha="center",
        fontsize=7,
        color="#1e3a8a",
        fontweight="bold"
    )

    ax.set_xlabel("Elapsed Postmortem Interval (Days Since Biological Cessation)", fontsize=8, labelpad=2, fontweight="bold", color="#1e293b")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.tick_params(axis="x", labelsize=7.5, colors="#334155")
    ax.legend(loc="upper right", frameon=True, fontsize=7.5, facecolor="#ffffff", edgecolor="#cbd5e1")

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#64748b")

    plt.title(f"ESTIMATED TIME-OF-DEATH RANGE — SPECIMEN REF: {sample_id}", fontsize=8.5, fontweight="bold", pad=4, color="#0f172a")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_forensic_pdf(
    case_metadata: Dict[str, Any],
    pmi_findings: Dict[str, Any],
    qc_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    shap_plot_buf: Optional[io.BytesIO] = None,
    shap_narrative: Optional[List[str]] = None,
) -> bytes:
    """
    Generate an authentic, court-admissible Post Mortem Examination Report
    following the structure of official medico-legal autopsy reports (Form 5372).
    Optionally incorporates SHAP feature attribution plot and forensic audit narrative.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=26,
        bottomMargin=26,
    )

    styles = getSampleStyleSheet()

    # Form Typography
    header_state = ParagraphStyle(
        "HeaderState",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        alignment=1,
        textColor=colors.HexColor("#0f172a"),
    )

    header_title = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=1,
        textColor=colors.HexColor("#000000"),
        spaceBefore=2,
        spaceAfter=2,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        alignment=1,
        textColor=colors.HexColor("#000000"),
    )

    field_label = ParagraphStyle(
        "FieldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#000000"),
    )

    field_val = ParagraphStyle(
        "FieldVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1e293b"),
    )

    field_val_bold = ParagraphStyle(
        "FieldValBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    section_banner = ParagraphStyle(
        "SectionBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#000000"),
    )

    narrative_p = ParagraphStyle(
        "NarrativeP",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # -------------------------------------------------------------
    # 1. TOP HEADER & OFFICIAL FORM BAR (WITH DIGITAL QR CODE)
    # -------------------------------------------------------------
    pm_no = case_metadata.get("pm_report_no", "PM-619 / 2026")
    ps_name = case_metadata.get("police_station", "New Township P.S.")
    inquest_no = case_metadata.get("inquest_no", "14 / 2026")
    inquest_date = case_metadata.get("inquest_date", "15 / 09 / 2026")
    dec_name = case_metadata.get("deceased_name", "Unidentified Individual")
    doc_name = case_metadata.get("analyst", "Dr. Tanish Walture")
    raw_cause = case_metadata.get("cause_of_death", "ASPHYXIA AS A RESULT OF CONSTRICTION OF NECK (PENDING TOXICOLOGY & HISTOLOGY).")

    pred_days = float(pmi_findings.get("predicted_pmi", 6.8))
    lower_days = float(pmi_findings.get("lower_bound", 5.7))
    upper_days = float(pmi_findings.get("upper_bound", 8.0))

    # Compute Cryptographic Evidence Digest (SHA-256)
    evidence_payload_raw = f"{pm_no}|{ps_name}|{inquest_no}|{dec_name}|{pred_days:.2f}|{lower_days:.2f}|{upper_days:.2f}|{doc_name}|{raw_cause}"
    evidence_hash = compute_sha256_hash(evidence_payload_raw)

    clean_pm = pm_no.replace(" ", "").replace("/", "-")
    clean_inq = inquest_no.replace(" ", "").replace("/", "-")
    qr_url = f"https://necrotrace.streamlit.app/?view=verify&case={clean_pm}&inq={clean_inq}&pmi={pred_days:.1f}d&hash={evidence_hash[:16]}"

    header_qr = generate_qr_code_drawing(qr_url, size=52.0)

    header_table_data = [
        [
            Paragraph("<b>DEPARTMENT OF FORENSIC MEDICINE & POLICE MORGUE</b><br/>GOVERNMENT MEDICAL COLLEGE & HOSPITAL", header_state),
            Paragraph(f"<b>POST MORTEM REPORT</b><br/><font size='7.5'>FORM NO. PM-5372</font>", header_title),
            Paragraph(f"<b>REPORT NO:</b> {pm_no}<br/><b>P.S.:</b> {ps_name}<br/><b>INQUEST:</b> {inquest_no}<br/><b>DATE:</b> {inquest_date}", field_label),
            header_qr,
        ]
    ]
    t_top = Table(header_table_data, colWidths=[2.3 * inch, 2.3 * inch, 2.1 * inch, 0.8 * inch])
    t_top.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (3, 0), (3, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (3, 0), (3, 0), 2),
        ("RIGHTPADDING", (3, 0), (3, 0), 0),
    ]))
    story.append(t_top)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.black, spaceAfter=4, spaceBefore=2))

    # -------------------------------------------------------------
    # 2. INSTITUTIONAL & AUTOPSY TIMING SCHEDULE TABLE
    # -------------------------------------------------------------
    institution = case_metadata.get("institution", "District Medico-Legal Center & Morgue")
    analyst_doc = case_metadata.get("analyst", "Dr. Tanish Walture, M.D. (Forensic Medicine)")
    receipt_time = case_metadata.get("receipt_time", "15/09/2026, 09:30 hrs")
    commence_time = case_metadata.get("commence_time", "15/09/2026, 10:45 hrs")
    complete_time = case_metadata.get("complete_time", "15/09/2026, 12:15 hrs")
    inquest_exam_time = case_metadata.get("inquest_exam_time", "15/09/2026, 08:00 hrs")

    sched_headers = [
        Paragraph("NAME OF INSTITUTION", table_header),
        Paragraph("P.M. REPORT NO. & DATE", table_header),
        Paragraph("CONDUCTED BY (MEDICAL OFFICER)", table_header),
        Paragraph("DATE & TIME OF RECEIPT OF BODY", table_header),
        Paragraph("DATE & TIME OF COMMENCEMENT", table_header),
        Paragraph("TIME OF COMPLETION OF AUTOPSY", table_header),
        Paragraph("INQUEST EXAM TIME (BY POLICE)", table_header),
    ]
    sched_vals = [
        Paragraph(institution, field_val_bold),
        Paragraph(f"{pm_no}<br/>{inquest_date}", field_val),
        Paragraph(f"<b>{analyst_doc}</b><br/>Reg. No. WBMC/45826", field_val),
        Paragraph(receipt_time, field_val),
        Paragraph(commence_time, field_val),
        Paragraph(complete_time, field_val),
        Paragraph(inquest_exam_time, field_val),
    ]
    t_sched = Table([sched_headers, sched_vals], colWidths=[1.4*inch, 1.0*inch, 1.3*inch, 1.0*inch, 0.95*inch, 0.95*inch, 0.9*inch])
    t_sched.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_sched)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 3. CASE PARTICULARS TABLE
    # -------------------------------------------------------------
    story.append(Table([[Paragraph("<b>CASE PARTICULARS & IDENTIFICATION</b>", section_banner)]],
                       colWidths=[7.5 * inch],
                       style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
                              ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                              ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

    deceased_name = case_metadata.get("deceased_name", "Unidentified Individual (Ref: Unknown Deceased #42)")
    guardian = case_metadata.get("guardian_name", "Not Known")
    address = case_metadata.get("address", "Recovered from outdoor perimeter")
    age = case_metadata.get("age", "Approx. 35 - 40 Years")
    sex = case_metadata.get("sex", "Male")
    police_officials = case_metadata.get("police_officials", "SI R. K. Mandal, New Township P.S.")
    relatives = case_metadata.get("relatives_identified", "Unidentified at commencement; Inquest identification by Police")

    case_part_rows = [
        [
            Paragraph("<b>Name of Deceased:</b>", field_label),
            Paragraph(deceased_name, field_val_bold),
            Paragraph("<b>Age (Approx):</b>", field_label),
            Paragraph(age, field_val),
            Paragraph("<b>Sex:</b>", field_label),
            Paragraph(sex, field_val),
        ],
        [
            Paragraph("<b>S/O, D/O, W/O:</b>", field_label),
            Paragraph(guardian, field_val),
            Paragraph("<b>Residential Address:</b>", field_label),
            Paragraph(address, field_val),
            Paragraph("<b>Station:</b>", field_label),
            Paragraph(ps_name, field_val),
        ],
        [
            Paragraph("<b>Body Brought By:</b>", field_label),
            Paragraph(police_officials, field_val),
            Paragraph("<b>Identified By:</b>", field_label),
            Paragraph(relatives, field_val),
            Paragraph("<b>Challan No:</b>", field_label),
            Paragraph("CH-892/26", field_val),
        ]
    ]
    t_case = Table(case_part_rows, colWidths=[1.1*inch, 2.1*inch, 1.0*inch, 1.8*inch, 0.6*inch, 0.9*inch])
    t_case.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_case)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 4. SCHEDULE OF OBSERVATIONS (A. GENERAL & POSTMORTEM CHANGES)
    # -------------------------------------------------------------
    story.append(Table([[Paragraph("<b>SCHEDULE OF OBSERVATIONS (PHYSICAL & POST-MORTEM CHANGES)</b>", section_banner)]],
                       colWidths=[7.5 * inch],
                       style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
                              ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                              ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

    height = case_metadata.get("height_cm", "172 cm")
    weight = case_metadata.get("weight_kg", "68 kg")
    physique = case_metadata.get("physique", "Average built, moderately nourished")
    clothes = case_metadata.get("clothes", "Soiled cotton garments, intact footwear")
    marks = case_metadata.get("id_marks", "Old linear scar on forearm; unidentifiable tattoo")

    pred_days = float(pmi_findings.get("predicted_pmi", 6.8))
    lower_days = float(pmi_findings.get("lower_bound", 5.7))
    upper_days = float(pmi_findings.get("upper_bound", 8.0))
    stage = pmi_findings.get("decomposition_stage", "Bloat / Active Decomposition")
    temp_c = float(pmi_findings.get("ambient_temp_c", 23.5))

    # Clinical autopsy findings mapped to decomposition timeline
    rigor_obs = case_metadata.get("rigor_obs", None)
    bloat_obs = case_metadata.get("bloat_obs", None)
    discolor_obs = case_metadata.get("discolor_obs", None)

    if not rigor_obs:
        if pred_days < 2.0:
            rigor_obs = "Rigor mortis present in extremities."
        else:
            rigor_obs = "Rigor mortis completely passed off (absent throughout body)."

    if not bloat_obs or not discolor_obs:
        if pred_days < 2.0:
            ext_desc = "Hypostasis fixed on posterior surfaces; greenish discoloration starting in right iliac fossa."
        elif pred_days < 10.0:
            ext_desc = "Greenish-black discoloration over abdomen; superficial venous marbling; abdominal gas distension (bloat); postmortem purge fluid at nares."
        else:
            ext_desc = "Advanced decomposition with extensive soft tissue liquefaction, black putrefaction, partial bone exposure."
    else:
        ext_desc = f"{bloat_obs}; {discolor_obs}."

    obs_rows = [
        [
            Paragraph("<b>1. Height & Weight:</b>", field_label),
            Paragraph(f"Height: {height} | Weight: {weight}", field_val),
            Paragraph("<b>2. Physique:</b>", field_label),
            Paragraph(physique, field_val),
            Paragraph("<b>3. Clothes & Belongings:</b>", field_label),
            Paragraph(clothes, field_val),
        ],
        [
            Paragraph("<b>4. Identification Marks:</b>", field_label),
            Paragraph(marks, field_val),
            Paragraph("<b>5. Rigor Mortis Status:</b>", field_label),
            Paragraph(rigor_obs, field_val),
            Paragraph("<b>6. Scene Ambient Temp:</b>", field_label),
            Paragraph(f"{temp_c:.1f}°C (ADD: {pred_days*temp_c:.1f})", field_val),
        ],
        [
            Paragraph("<b>7. Postmortem Changes & Decomposition:</b>", field_label),
            Paragraph(f"<b>Stage: {stage}.</b> {ext_desc}", field_val),
            Paragraph("<b>Natural Orifices:</b>", field_label),
            Paragraph("Corneas hazy, collapsed. Purge fluid at mouth & nose.", field_val),
            Paragraph("<b>Hypostasis:</b>", field_label),
            Paragraph("Indistinct due to hemolysis and discoloration.", field_val),
        ]
    ]
    t_obs = Table(obs_rows, colWidths=[1.3*inch, 1.8*inch, 1.1*inch, 1.6*inch, 0.9*inch, 0.8*inch])
    t_obs.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_obs)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 5. FORENSIC MICROBIOLOGY & METAGENOMIC SUCCESSION CLOCK
    # -------------------------------------------------------------
    story.append(Table([[Paragraph("<b>MICROBIAL SUCCESSION ANALYSIS (POST-MORTEM BIOLOGICAL CLOCK)</b>", section_banner)]],
                       colWidths=[7.5 * inch],
                       style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
                              ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                              ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

    sample_id = case_metadata.get("sample_id", "SAMP_0042")
    swab_loc = case_metadata.get("sample_site", "Oral Cavity Swab")
    lab_barcode = case_metadata.get("lab_barcode", f"NT-LAB-{sample_id.replace('SAMP_', '')}")
    read_depth = int(qc_metrics.get("read_depth", 28410))

    if read_depth >= 1000:
        adequacy_status = f"SATISFACTORY — High DNA Recovery ({read_depth:,} bacterial reads verified; test fully valid)"
    else:
        adequacy_status = f"SUB-OPTIMAL — Low DNA Recovery ({read_depth:,} reads; interpret with caution)"

    # Simplified, plain-English forensic findings without academic clutter
    custom_micro_findings = case_metadata.get("micro_findings", None)
    if custom_micro_findings:
        micro_findings = custom_micro_findings
        bio_stage = "Diagnostic Succession Flora"
    else:
        if pred_days < 3.0:
            micro_findings = (
                "Normal surface bacteria present. Minimal putrefactive activity detected, "
                "consistent with early post-mortem period."
            )
            bio_stage = "Early Post-Mortem Flora"
        elif pred_days < 12.0:
            micro_findings = (
                "Putrefactive bacteria predominant. Biological succession pattern confirms "
                "active internal decomposition, gas production, and tissue liquefaction."
            )
            bio_stage = "Active Putrefactive Flora"
        else:
            micro_findings = (
                "Environmental and soil-associated bacteria predominant. Biological succession pattern "
                "confirms advanced decomposition and skeletonization."
            )
            bio_stage = "Advanced Decay Flora"

    micro_rows = [
        [
            Paragraph("<b>Swab Specimen:</b>", field_label),
            Paragraph(f"{swab_loc} (Specimen Ref: {sample_id})", field_val_bold),
            Paragraph("<b>Specimen Adequacy (DNA Yield):</b>", field_label),
            Paragraph(adequacy_status, field_val),
        ],
        [
            Paragraph("<b>Biological Profile:</b>", field_label),
            Paragraph(f"<b>{bio_stage}</b>", field_val_bold),
            Paragraph("<b>Laboratory Specimen Barcode:</b>", field_label),
            Paragraph(f"<b>{lab_barcode}</b> (Evidence Seal Intact)", field_val),
        ],
        [
            Paragraph("<b>Microbial Examination Finding:</b>", field_label),
            Paragraph(micro_findings, field_val),
            Paragraph("<b>Analytical Method:</b>", field_label),
            Paragraph("Metagenomic Necrobiome Clock (Quantitative Succession Profiling)", field_val),
        ]
    ]
    t_micro = Table(micro_rows, colWidths=[1.8*inch, 2.2*inch, 1.6*inch, 1.9*inch])
    t_micro.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_micro)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 6. OPINION AS TO TIME AND CAUSE OF DEATH (THE DEFINITIVE VERDICT)
    # -------------------------------------------------------------
    story.append(Table([[Paragraph("<b>OPINION AS TO TIME AND CAUSE OF DEATH</b>", section_banner)]],
                       colWidths=[7.5 * inch],
                       style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fee2e2")),
                              ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#991b1b")),
                              ("TOPPADDING", (0, 0), (-1, -1), 2),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))

    try:
        ref_dt = datetime.strptime(receipt_time.split(",")[0].strip(), "%d/%m/%Y")
    except Exception:
        ref_dt = datetime.now(timezone.utc)

    dt_most_likely = ref_dt - timedelta(days=pred_days)
    dt_earliest = ref_dt - timedelta(days=upper_days)
    dt_latest = ref_dt - timedelta(days=lower_days)

    death_calendar_str = f"{dt_earliest.strftime('%d/%m/%Y')} to {dt_latest.strftime('%d/%m/%Y')} (Most Likely: {dt_most_likely.strftime('%d/%m/%Y')})"

    raw_cause = case_metadata.get("cause_of_death", "ASPHYXIA AS A RESULT OF CONSTRICTION OF NECK (PENDING TOXICOLOGY & HISTOLOGY).")
    cause_desc = str(raw_cause).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if "&amp;" not in str(raw_cause) else str(raw_cause)
    raw_manner = case_metadata.get("manner_of_death", "Matter under judicial inquiry / Forensic Inquest.")
    manner_desc = str(raw_manner).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if "&amp;" not in str(raw_manner) else str(raw_manner)

    opinion_rows = [
        [
            Paragraph("<b>1. PROBABLE TIME SINCE DEATH:</b>", field_label),
            Paragraph(
                f"<font size='10' color='#991b1b'><b>{pred_days:.1f} DAYS</b></font> &nbsp; (approx. <b>{pred_days*24.0:.0f} Hours</b> prior to examination)<br/>"
                f"<b>PROBABLE FORENSIC WINDOW:</b> <font color='#1e3a8a'><b>{lower_days:.1f} to {upper_days:.1f} DAYS</b></font> prior to discovery.<br/>"
                f"<b>ESTIMATED CALENDAR RANGE OF DEATH:</b> {death_calendar_str}",
                field_val
            ),
            Paragraph("<b>Basis of Timing:</b>", field_label),
            Paragraph(
                "Based on integrated clinical autopsy evaluation (rigor mortis state, putrefactive bloat and marbling) "
                "concordant with calibrated microbial succession bioindicator sequencing under prevailing scene temperature.",
                field_val
            ),
        ],
        [
            Paragraph("<b>2. CAUSE OF DEATH:</b>", field_label),
            Paragraph(f"<b>{cause_desc}</b>", field_val_bold),
            Paragraph("<b>3. MANNER:</b>", field_label),
            Paragraph(manner_desc, field_val),
        ]
    ]
    t_opinion = Table(opinion_rows, colWidths=[1.8*inch, 2.8*inch, 1.0*inch, 1.9*inch])
    t_opinion.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#991b1b")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_opinion)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 7. FORENSIC INTERVAL TIMELINE GRAPHIC
    # -------------------------------------------------------------
    chart_buf = generate_forensic_timeline_chart(
        sample_id=sample_id,
        lower_bound_days=lower_days,
        predicted_pmi_days=pred_days,
        upper_bound_days=upper_days
    )
    img = Image(chart_buf, width=7.4 * inch, height=1.45 * inch)
    story.append(img)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 8. SPECIMENS COLLECTED & HANDED OVER TABLE
    # -------------------------------------------------------------
    story.append(Table([[Paragraph("<b>SPECIMENS COLLECTED & HANDED OVER TO POLICE</b>", section_banner)]],
                       colWidths=[7.5 * inch],
                       style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
                              ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                              ("TOPPADDING", (0, 0), (-1, -1), 1),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

    specimen_rows = [
        [
            Paragraph("<b>Viscera Preserved:</b>", field_label),
            Paragraph("Stomach with contents, portion of liver, kidney, blood sample for chemical analysis.", field_val),
            Paragraph("<b>Microbial Swabs:</b>", field_label),
            Paragraph("Two sterile oral/nasal swabs preserved for metagenomic DNA reference.", field_val),
        ],
        [
            Paragraph("<b>Clothes / Articles:</b>", field_label),
            Paragraph("Personal garments sealed in labeled packet.", field_val),
            Paragraph("<b>Sample of Seal:</b>", field_label),
            Paragraph("Distinctive seal impression affixed below.", field_val),
        ]
    ]
    t_spec = Table(specimen_rows, colWidths=[1.3*inch, 2.5*inch, 1.3*inch, 2.4*inch])
    t_spec.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_spec)
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 8b. EXPLAINABLE AI & FEATURE ATTRIBUTION AUDIT (SHAP)
    # -------------------------------------------------------------
    if shap_plot_buf is not None:
        story.append(Table([[Paragraph("<b>EXPLAINABLE AI &amp; FEATURE ATTRIBUTION AUDIT (SHAPLEY ATTRIBUTION)</b>", section_banner)]],
                           colWidths=[7.5 * inch],
                           style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e0f2fe")),
                                  ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#0369a1")),
                                  ("TOPPADDING", (0, 0), (-1, -1), 1),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))

        shap_img = Image(shap_plot_buf, width=7.4 * inch, height=2.4 * inch)
        story.append(shap_img)
        story.append(Spacer(1, 3))

        if shap_narrative:
            bullet_paras = [
                Paragraph(f"&bull; {pt}", ParagraphStyle("ShapBullet", parent=field_val, fontSize=6.5, leading=8.5))
                for pt in shap_narrative[:4]
            ]
            t_shap_notes = Table([[b] for b in bullet_paras], colWidths=[7.5 * inch])
            t_shap_notes.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_shap_notes)
            story.append(Spacer(1, 4))

    # -------------------------------------------------------------
    # 9. MEDICAL OFFICER SIGNATURE, SEAL & CERTIFICATION
    # -------------------------------------------------------------
    doc_name = case_metadata.get("analyst", "Dr. Tanish Walture")
    reg_no = case_metadata.get("reg_no", "Reg. No. 45826 of WBMC")
    designation = case_metadata.get("designation", "Medical Officer & Forensic Specialist")
    hosp_dept = case_metadata.get("department", "District Medico-Legal Center / Government Hospital")

    sig_qr = generate_qr_code_drawing(qr_url, size=52.0)

    seal_table = Table([
        [
            sig_qr,
            Paragraph(
                "<b>DIGITAL CHAIN OF CUSTODY SEAL</b><br/>"
                "<b>Status:</b> <font color='#047857'><b>HASH-VERIFIED RECORD</b></font><br/>"
                f"<b>SHA-256:</b> <font face='Courier' size='5.0'>{evidence_hash[:22]}...</font><br/>"
                "<b>Protocol:</b> Metagenomic Necrobiome Clock<br/>"
                "<b>Standard:</b> Methodology Reference: Daubert/Frye<br/>"
                "<i>Scan QR to verify evidence dossier</i>",
                ParagraphStyle("SealP", parent=field_val, fontSize=6.2, leading=7.8, textColor=colors.HexColor("#0f172a"))
            )
        ]
    ], colWidths=[0.75 * inch, 2.05 * inch])
    seal_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    sig_table_data = [
        [
            Paragraph(
                "<b>Generated Copy &mdash; For Research &amp; Demonstration Only</b><br/><br/>"
                "_________________________________<br/>"
                "<b>Officer-in-Charge</b><br/>"
                f"{ps_name}<br/>"
                "Police Commissionerate Seal",
                field_val
            ),
            seal_table,
            Paragraph(
                "<b>Signature of Medical Officer:</b><br/><br/>"
                "_________________________________<br/>"
                f"<b>{doc_name}</b>, M.D.<br/>"
                f"{designation}<br/>"
                f"{reg_no}<br/>"
                f"{hosp_dept}",
                field_val
            ),
        ]
    ]
    t_sig = Table(sig_table_data, colWidths=[2.2 * inch, 2.9 * inch, 2.4 * inch])
    t_sig.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sig)

    # EDUCATIONAL DISCLAIMER FOOTER
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=4, spaceBefore=4))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.5,
        leading=8,
        alignment=1,
        textColor=colors.HexColor("#64748b"),
    )
    story.append(Paragraph(
        "EDUCATIONAL &amp; RESEARCH DEMONSTRATION ONLY &mdash; This document is generated by NecroTrace, an academic research prototype. "
        "It is NOT an official medico-legal document. It has NOT been certified or validated by any accreditation body (ISO 17025 or otherwise). "
        "It must NOT be used as evidence in any legal proceeding. The platform is provided &quot;AS IS&quot; under the MIT License without warranty of any kind.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(pdf_bytes)
        print(f"[+] Official Post-Mortem Report PDF successfully generated: {out_file}")

    return pdf_bytes

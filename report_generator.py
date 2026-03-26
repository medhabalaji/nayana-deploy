from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import io
import os
 
from constants import DISEASE_NAMES
DISEASE_COLORS_HEX = [
    '#2d9e6b','#e63946','#f4a261','#457b9d',
    '#9b5de5','#f77f00','#00b4d8','#74c69d'
]
 
GREEN      = colors.HexColor('#2d9e6b')
DARK_BLUE  = colors.HexColor('#1e293b')
MID_GRAY   = colors.HexColor('#64748b')
LIGHT_GRAY = colors.HexColor('#f1f5f9')
RED        = colors.HexColor('#e63946')
AMBER      = colors.HexColor('#f4a261')
WHITE      = colors.white
 
DISEASE_INFO = {
    'Diabetic Retinopathy': {
        'what': 'Damage to blood vessels in the retina caused by high blood sugar over time.',
        'symptoms': 'Blurred vision, floaters, dark spots, difficulty seeing at night.',
        'urgency': 'See an ophthalmologist within 1-2 weeks.',
        'tip': 'Keeping blood sugar levels controlled can slow or prevent progression.',
        'serious': True
    },
    'Glaucoma': {
        'what': 'A group of eye conditions that damage the optic nerve, often due to high eye pressure.',
        'symptoms': 'Gradual peripheral vision loss, tunnel vision in advanced stages.',
        'urgency': 'See an ophthalmologist within 1 week. Early treatment prevents blindness.',
        'tip': 'Glaucoma often has no symptoms until vision loss has occurred. Regular screening is key.',
        'serious': True
    },
    'Cataract': {
        'what': 'Clouding of the normally clear lens of the eye.',
        'symptoms': 'Blurry vision, faded colours, glare, poor night vision.',
        'urgency': 'Schedule a detailed eye examination. Surgery is highly effective.',
        'tip': 'Cataracts are very treatable — most patients regain excellent vision after surgery.',
        'serious': False
    },
    'AMD': {
        'what': 'Age-related Macular Degeneration — deterioration of the central part of the retina.',
        'symptoms': 'Blurred or distorted central vision, difficulty reading, blind spots.',
        'urgency': 'Urgent referral to a retina specialist. Some forms are treatable if caught early.',
        'tip': 'Smoking significantly increases AMD risk. A diet rich in leafy greens may help.',
        'serious': True
    },
    'Hypertension': {
        'what': 'High blood pressure can damage blood vessels in the retina.',
        'symptoms': 'Often no eye symptoms until advanced. May cause blurred vision or vision loss.',
        'urgency': 'Consult a physician to manage blood pressure. Eye review in 1-3 months.',
        'tip': 'Controlling blood pressure protects both your eyes and your heart.',
        'serious': False
    },
    'Myopia': {
        'what': 'Short-sightedness — difficulty seeing distant objects clearly.',
        'symptoms': 'Distant objects appear blurry, squinting, headaches.',
        'urgency': 'Schedule a routine eye examination for prescription update.',
        'tip': 'High myopia increases risk of retinal detachment. Regular monitoring recommended.',
        'serious': False
    },
}
 
def _styles():
    base = getSampleStyleSheet()
    return {
        'cover_title': ParagraphStyle('CoverTitle', parent=base['Normal'],
            fontSize=28, fontName='Helvetica-Bold',
            textColor=GREEN, alignment=TA_CENTER, spaceAfter=4),
        'cover_sub': ParagraphStyle('CoverSub', parent=base['Normal'],
            fontSize=13, fontName='Helvetica',
            textColor=DARK_BLUE, alignment=TA_CENTER, spaceAfter=4),
        'cover_body': ParagraphStyle('CoverBody', parent=base['Normal'],
            fontSize=10, fontName='Helvetica',
            textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=3),
        'section': ParagraphStyle('Section', parent=base['Normal'],
            fontSize=12, fontName='Helvetica-Bold',
            textColor=DARK_BLUE, spaceBefore=12, spaceAfter=6),
        'body': ParagraphStyle('Body', parent=base['Normal'],
            fontSize=9, fontName='Helvetica',
            textColor=DARK_BLUE, spaceAfter=3, leading=13),
        'small': ParagraphStyle('Small', parent=base['Normal'],
            fontSize=8, fontName='Helvetica',
            textColor=MID_GRAY, alignment=TA_CENTER),
        'caption': ParagraphStyle('Caption', parent=base['Normal'],
            fontSize=8, fontName='Helvetica-Oblique',
            textColor=MID_GRAY, alignment=TA_CENTER),
        'risk_high': ParagraphStyle('RiskHigh', parent=base['Normal'],
            fontSize=13, fontName='Helvetica-Bold',
            textColor=RED, alignment=TA_CENTER),
        'risk_mod': ParagraphStyle('RiskMod', parent=base['Normal'],
            fontSize=13, fontName='Helvetica-Bold',
            textColor=AMBER, alignment=TA_CENTER),
        'risk_low': ParagraphStyle('RiskLow', parent=base['Normal'],
            fontSize=13, fontName='Helvetica-Bold',
            textColor=GREEN, alignment=TA_CENTER),
        'label': ParagraphStyle('Label', parent=base['Normal'],
            fontSize=9, fontName='Helvetica-Bold',
            textColor=MID_GRAY, spaceAfter=2),
        'note': ParagraphStyle('Note', parent=base['Normal'],
            fontSize=9, fontName='Helvetica-Oblique',
            textColor=MID_GRAY, spaceAfter=3, leading=13),
    }
 
def _hr(story, color=None, thickness=1):
    story.append(HRFlowable(
        width="100%", thickness=thickness,
        color=color or colors.HexColor('#e2e8f0'),
        spaceAfter=8, spaceBefore=4
    ))
 
def _section(story, title, s):
    story.append(Paragraph(title, s['section']))
    _hr(story, GREEN, 0.5)
 
def _info_row(label, value, s):
    return [Paragraph(label, s['label']), Paragraph(str(value), s['body'])]
 
def _pil_to_buf(pil_img, size=(300, 300)):
    buf = io.BytesIO()
    pil_img.resize(size).save(buf, format='PNG')
    buf.seek(0)
    return buf
 
def _np_to_buf(arr):
    buf = io.BytesIO()
    plt.imsave(buf, arr.astype(np.uint8), format='png')
    buf.seek(0)
    return buf
 
def _bar_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    bars = ax.barh(DISEASE_NAMES, [p * 100 for p in probs],
                   color=DISEASE_COLORS_HEX, height=0.55, edgecolor='none')
    ax.set_xlabel("Confidence (%)", fontsize=9, color='#64748b')
    ax.set_xlim(0, 108)
    ax.axvline(50, color='#cbd5e1', lw=0.8, ls='--')
    ax.tick_params(colors='#64748b', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#e2e8f0')
    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{p * 100:.1f}%', va='center', fontsize=7.5, color='#64748b')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
 
def _trend_chart(visit_history):
    if not visit_history or len(visit_history) < 2:
        return None
    def rs(r):
        r = r.lower()
        if "high" in r or "specialist" in r: return 3
        elif "moderate" in r or "follow" in r: return 2
        return 1
    dates  = [v['timestamp'][:6] for v in visit_history]
    scores = [rs(v['risk_level']) for v in visit_history]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    clrs = {1: '#2d9e6b', 2: '#f4a261', 3: '#e63946'}
    ax.plot(dates, scores, color='#b7e4c7', linewidth=2, zorder=1)
    ax.scatter(dates, scores, c=[clrs[s] for s in scores], s=60, zorder=2)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Moderate', 'High'], fontsize=8, color='#64748b')
    ax.tick_params(axis='x', colors='#64748b', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#e2e8f0')
    ax.set_ylim(0.5, 3.5)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
 
def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(GREEN)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(2 * cm, h - 1.2 * cm, "NAYANA - CONFIDENTIAL EYE SCREENING REPORT")
    canvas.setFillColor(MID_GRAY)
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, datetime.now().strftime("%d %b %Y"))
    canvas.setStrokeColor(colors.HexColor('#e2e8f0'))
    canvas.line(2 * cm, h - 1.4 * cm, w - 2 * cm, h - 1.4 * cm)
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.setFillColor(MID_GRAY)
    canvas.setFont('Helvetica', 7)
    canvas.drawString(2 * cm, 0.9 * cm,
                      "This report is AI-assisted. Not a substitute for professional medical advice.")
    canvas.drawRightString(w - 2 * cm, 0.9 * cm, f"Page {doc.page}")
    canvas.restoreState()
 
def generate_report(
        patient_name, patient_age, patient_gender,
        symptoms, quality_score, quality_tips,
        probs, detected_conditions, risk_level,
        original_image_pil, heatmap_array,
        output_path="screening_report.pdf",
        language="English",
        patient_email="",
        patient_id="",
        symptoms_method="typed",
        voice_transcript="",
        voice_language="English",
        questionnaire_answers=None,
        triage_decision="front",
        front_eye_image_pil=None,
        front_eye_results=None,
        front_eye_recommendations=None,
        front_eye_quality=0,
        fundus_quality_tips=None,
        risk_type="low",
        doctor_name=None,
        doctor_diagnosis=None,
        doctor_prescription=None,
        doctor_referral=None,
        doctor_notes=None,
        reviewed_at=None,
        chat_messages=None,
        visit_history=None,
):
    if questionnaire_answers is None:
        questionnaire_answers = {}
    if front_eye_results is None:
        front_eye_results = {}
    if front_eye_recommendations is None:
        front_eye_recommendations = []
    if fundus_quality_tips is None:
        fundus_quality_tips = quality_tips or []
    if chat_messages is None:
        chat_messages = []
    if visit_history is None:
        visit_history = []
 
    # Keep only last 5 visits for clean history
    recent_visits = visit_history[-5:] if len(visit_history) > 5 else visit_history
 
    s = _styles()
    now_str   = datetime.now().strftime("%d %B %Y, %I:%M %p")
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not patient_id:
        patient_id = f"P-{abs(hash(patient_email)) % 99999:05d}" if patient_email else "P-00000"
 
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2 * cm
    )
    story = []
 
    # ══════════════════════════════════════════════════════════
    # PAGE 1 - COVER
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("nayana", s['cover_title']))
    story.append(Paragraph("the eye", s['cover_sub']))
    story.append(Spacer(1, 0.3 * cm))
    _hr(story, GREEN, 2)
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("CONFIDENTIAL EYE SCREENING REPORT", s['cover_sub']))
    story.append(Spacer(1, 1 * cm))
 
    cover_data = [
        ["Patient Name", patient_name,       "Report ID",  report_id],
        ["Patient ID",   patient_id,          "Date",       now_str],
        ["Age / Gender", f"{patient_age}y / {patient_gender}", "Screened by", "Nayana AI v1.0"],
        ["Email",        patient_email or "-", "Language",  language],
    ]
    cover_table = Table(cover_data, colWidths=[3 * cm, 5.5 * cm, 3.5 * cm, 5 * cm])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('BACKGROUND', (2, 0), (2, -1), LIGHT_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK_BLUE),
        ('TEXTCOLOR', (2, 0), (2, -1), DARK_BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 1.5 * cm))
 
    r_style = s['risk_high'] if 'high' in risk_type else s['risk_mod'] if 'moderate' in risk_type else s['risk_low']
    r_bg    = colors.HexColor('#fff0f0') if 'high' in risk_type else colors.HexColor('#fffbe6') if 'moderate' in risk_type else colors.HexColor('#f0fdf4')
    r_bc    = RED if 'high' in risk_type else AMBER if 'moderate' in risk_type else GREEN
    risk_label = "High Risk" if 'high' in risk_type else "Moderate Risk" if 'moderate' in risk_type else "Low Risk"
    risk_box = Table([[Paragraph(f"Overall Risk: {risk_label}", r_style)]], colWidths=[17 * cm])
    risk_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), r_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, r_bc),
        ('PADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(risk_box)
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        '"This report is AI-assisted and must be confirmed by a qualified ophthalmologist."',
        s['note']
    ))
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 2 - PATIENT PROFILE & SYMPTOMS
    # ══════════════════════════════════════════════════════════
    _section(story, "Patient Information", s)
    info_data = [
        _info_row("Full Name",  patient_name, s),
        _info_row("Age",        f"{patient_age} years", s),
        _info_row("Gender",     patient_gender, s),
        _info_row("Email",      patient_email or "Not provided", s),
        _info_row("Patient ID", patient_id, s),
    ]
    info_table = Table(info_data, colWidths=[4 * cm, 14 * cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4 * cm))
 
    _section(story, "Reported Symptoms", s)
    story.append(Paragraph(f"Method: {symptoms_method.title()}", s['body']))
    if voice_transcript:
        story.append(Paragraph(f"Voice transcript ({voice_language}): {voice_transcript}", s['note']))
 
    # Strip front-eye data from symptoms for clean display
    clean_symptoms = symptoms.split(" | Front-eye:")[0] if " | Front-eye:" in symptoms else symptoms
    story.append(Paragraph(f"Symptoms: {clean_symptoms or 'Not specified'}", s['body']))
    story.append(Spacer(1, 0.3 * cm))
 
    if questionnaire_answers:
        _section(story, "Symptom Questionnaire", s)
        q_data = [["Question", "Response"]]
        for q, ans in questionnaire_answers.items():
            q_data.append([q, "Yes" if ans else "No"])
        q_table = Table(q_data, colWidths=[12 * cm, 5 * cm])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ]))
        story.append(q_table)
        story.append(Spacer(1, 0.3 * cm))
 
    story.append(Paragraph(
        f"AI Triage Decision: {'Retinal scan recommended' if triage_decision == 'fundus' else 'Front-eye screening recommended'}",
        s['body']
    ))
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 3 - FRONT EYE ANALYSIS
    # ══════════════════════════════════════════════════════════
    _section(story, "Front Eye Analysis", s)
 
    if front_eye_image_pil and front_eye_results:
        fe_quality_display = front_eye_quality if front_eye_quality > 0 else quality_score
        story.append(Paragraph(f"Image Quality: {fe_quality_display}%", s['body']))
        story.append(Spacer(1, 0.3 * cm))
 
        fe_buf = _pil_to_buf(front_eye_image_pil, (250, 250))
        fe_img = Image(fe_buf, width=6 * cm, height=6 * cm)
 
        fe_data = [["Condition", "Confidence", "Status"]]
        for cond, conf in sorted(front_eye_results.items(), key=lambda x: x[1], reverse=True):
            status = "Detected" if conf > 0.6 else "Possible" if conf > 0.3 else "Normal"
            fe_data.append([cond, f"{conf * 100:.0f}%", status])
 
        fe_img.width  = 10 * cm
        fe_img.height = 10 * cm
        fe_img.hAlign = 'CENTER'
        story.append(fe_img)
        story.append(Spacer(1, 0.4 * cm))

        fe_table = Table(fe_data, colWidths=[10 * cm, 3.5 * cm, 3.5 * cm])
        fe_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ]))
        story.append(fe_table)
        story.append(Spacer(1, 0.3 * cm))
 
        if front_eye_recommendations:
            story.append(Paragraph("Recommendations:", s['label']))
            for rec in front_eye_recommendations:
                story.append(Paragraph(f"- {rec}", s['body']))
    else:
        story.append(Paragraph("No front eye photo was provided for this screening.", s['note']))
 
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 4 - RETINAL FUNDUS ANALYSIS
    # ══════════════════════════════════════════════════════════
    _section(story, "Retinal Fundus Analysis", s)
 
    if original_image_pil is not None:
        story.append(Paragraph(f"Image Quality Score: {quality_score}%", s['body']))
        if quality_tips:
            for tip in quality_tips:
                story.append(Paragraph(f"  Note: {tip}", s['note']))
        story.append(Spacer(1, 0.3 * cm))
 
        orig_buf = _pil_to_buf(original_image_pil, (280, 280))
        orig_img = Image(orig_buf, width=6.5 * cm, height=6.5 * cm)
 
        orig_img.width  = 11 * cm
        orig_img.height = 11 * cm
        orig_img.hAlign = 'CENTER'
        story.append(orig_img)
        story.append(Paragraph("Original Fundus Image", s['caption']))
        story.append(Spacer(1, 0.6 * cm))

        if heatmap_array is not None:
            heat_buf = _np_to_buf(heatmap_array)
            heat_img = Image(heat_buf)
            heat_img.width  = 11 * cm
            heat_img.height = 11 * cm
            heat_img.hAlign = 'CENTER'
            story.append(heat_img)
            story.append(Paragraph("GradCAM AI Attention Map (Red/yellow = areas of focus)", s['caption']))
            story.append(Spacer(1, 0.4 * cm))
        story.append(Spacer(1, 0.4 * cm))
 
        chart_buf = _bar_chart(probs)
        chart_img = Image(chart_buf, width=17 * cm, height=8 * cm)
        story.append(chart_img)
        story.append(Spacer(1, 0.3 * cm))
 
        pred_data = [["Disease", "Confidence", "Flag"]]
        for i, name in enumerate(DISEASE_NAMES):
            p    = probs[i]
            flag = "HIGH" if p > 0.7 else "MODERATE" if p > 0.5 else "LOW" if p > 0.3 else "-"
            pred_data.append([name, f"{p * 100:.1f}%", flag])
 
        pred_table = Table(pred_data, colWidths=[8 * cm, 4 * cm, 5 * cm])
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]
        for i, p in enumerate(probs, 1):
            if p > 0.7:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fff0f0')))
            elif p > 0.5:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fffbe6')))
            else:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), WHITE if i % 2 == 0 else LIGHT_GRAY))
        pred_table.setStyle(TableStyle(style_cmds))
        story.append(pred_table)
    else:
        story.append(Paragraph("No retinal scan was provided for this screening.", s['note']))
 
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 5 - DISEASE INFORMATION
    # ══════════════════════════════════════════════════════════
    _section(story, "What These Findings Mean", s)
    story.append(Paragraph("Plain-language explanations for the detected conditions.", s['note']))
    story.append(Spacer(1, 0.2 * cm))
 
    non_normal = []
    if probs is not None:
        non_normal = [(DISEASE_NAMES[i], probs[i]) for i in range(1, 8) if probs[i] > 0.3]
 
    if non_normal:
        for dname, dp in sorted(non_normal, key=lambda x: x[1], reverse=True):
            info = DISEASE_INFO.get(dname)
            if not info:
                continue
            story.append(Paragraph(f"{dname} ({dp * 100:.0f}% confidence)", s['section']))
            d_data = [
                ["What is it?",   info['what']],
                ["Symptoms",      info['symptoms']],
                ["What to do",    info['urgency']],
                ["Important tip", info['tip']],
            ]
            d_table = Table(d_data, colWidths=[3.5 * cm, 13.5 * cm])
            d_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (1, 0), (1, -1), [WHITE, LIGHT_GRAY, WHITE, LIGHT_GRAY]),
            ]))
            story.append(d_table)
            story.append(Spacer(1, 0.4 * cm))
    else:
        story.append(Paragraph("No significant conditions detected above 30% confidence threshold.", s['body']))
 
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 6 - DOCTOR'S REVIEW
    # ══════════════════════════════════════════════════════════
    _section(story, "Specialist Review", s)
 
    if doctor_diagnosis:
        dr_data = [
            _info_row("Reviewed by",  doctor_name or "-", s),
            _info_row("Review date",  reviewed_at or "-", s),
            _info_row("Diagnosis",    doctor_diagnosis, s),
            _info_row("Treatment",    doctor_prescription or "None", s),
            _info_row("Referral",     doctor_referral or "-", s),
        ]
        if doctor_notes:
            dr_data.append(_info_row("Notes", doctor_notes, s))
        dr_table = Table(dr_data, colWidths=[4 * cm, 13 * cm])
        dr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(dr_table)
    else:
        story.append(Paragraph("This case is currently awaiting specialist review.", s['note']))
 
    if chat_messages:
        story.append(Spacer(1, 0.4 * cm))
        _section(story, "Messages", s)
        for msg in chat_messages:
            role = msg.get('sender_role', '').title()
            name = msg.get('sender_name', '')
            ts   = msg.get('timestamp', '')
            text = msg.get('text', '')
            story.append(Paragraph(f"{name} ({role}) - {ts}", s['label']))
            story.append(Paragraph(text, s['body']))
            story.append(Spacer(1, 0.2 * cm))
 
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 7 - VISIT HISTORY
    # ══════════════════════════════════════════════════════════
    _section(story, "Previous Screening History", s)
 
    if recent_visits and len(recent_visits) > 1:
        hist_data = [["Visit", "Date", "Risk", "Top Finding", "Confidence"]]
        for i, v in enumerate(recent_visits):
            det   = v.get('detected_conditions', [])
            top   = max(det, key=lambda x: x[1]) if det else ("Normal", 0)
            risk  = v['risk_level']
            r     = risk.lower()
            lvl   = "High" if ("high" in r or "specialist" in r) else "Moderate" if ("moderate" in r or "follow" in r) else "Low"
            label = "Current" if i == len(recent_visits) - 1 else f"Visit {i + 1}"
            hist_data.append([label, v['timestamp'][:11], lvl, top[0], f"{top[1] * 100:.0f}%"])
 
        hist_table = Table(hist_data, colWidths=[2.5 * cm, 3.5 * cm, 2.5 * cm, 6 * cm, 2.5 * cm])
        hist_style = [
            ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ('BACKGROUND', (0, len(recent_visits)), (-1, len(recent_visits)), colors.HexColor('#f0fdf4')),
            ('FONTNAME', (0, len(recent_visits)), (-1, len(recent_visits)), 'Helvetica-Bold'),
        ]
        hist_table.setStyle(TableStyle(hist_style))
        story.append(hist_table)
        story.append(Spacer(1, 0.4 * cm))
 
        trend_buf = _trend_chart(recent_visits)
        if trend_buf:
            story.append(Paragraph("Risk Trend Over Time", s['label']))
            trend_img = Image(trend_buf, width=17 * cm, height=6 * cm)
            story.append(trend_img)
    else:
        story.append(Paragraph("This is the patient's first screening. No previous history available.", s['note']))
 
    story.append(PageBreak())
 
    # ══════════════════════════════════════════════════════════
    # PAGE 8 - DISCLAIMER & FOOTER
    # ══════════════════════════════════════════════════════════
    _section(story, "Report Information", s)
    footer_data = [
        _info_row("Report ID",  report_id, s),
        _info_row("Generated",  now_str, s),
        _info_row("Platform",   "Nayana AI Tele-Ophthalmology v1.0", s),
        _info_row("AI Model",   "EfficientNet-B0 - ODIR-5K (8 diseases)", s),
        _info_row("Eye Model",  "EfficientNet-B0 - External Eye (6 conditions)", s),
    ]
    footer_table = Table(footer_data, colWidths=[3.5 * cm, 13.5 * cm])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(footer_table)
    story.append(Spacer(1, 0.6 * cm))
 
    _section(story, "Disclaimer", s)
    story.append(Paragraph(
        "This report is generated by an AI screening system and is intended for informational "
        "and preliminary screening purposes only. It does NOT constitute a medical diagnosis. "
        "All findings must be reviewed and confirmed by a qualified ophthalmologist before any "
        "clinical or treatment decisions are made. In case of high-risk findings, please refer "
        "to a specialist immediately. Nayana AI is not liable for any decisions made solely "
        "on the basis of this report.",
        s['body']
    ))
    story.append(Spacer(1, 0.6 * cm))
 
    _section(story, "Emergency Eye Care Contacts", s)
    contact_data = [
        ["Hospital", "Phone", "City"],
        ["Sankara Nethralaya",      "044-28281919", "Chennai"],
        ["Narayana Nethralaya",     "080-66121900", "Bengaluru"],
        ["LV Prasad Eye Institute", "040-30612612", "Hyderabad"],
        ["Aravind Eye Hospital",    "0452-4356100", "Madurai"],
    ]
    contact_table = Table(contact_data, colWidths=[7 * cm, 5 * cm, 5 * cm])
    contact_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
    ]))
    story.append(contact_table)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("nayana", s['cover_title']))
    story.append(Paragraph("the eye", s['cover_sub']))
 
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path
 
def generate_prescription_pdf(
        patient_name, patient_age, patient_gender,
        case_id, doctor_diagnosis, doctor_prescription,
        doctor_referral, doctor_notes, risk_level, reviewed_at,
        output_path="prescription.pdf"):
    s = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2 * cm
    )
    story = []
    
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("nayana", s['cover_title']))
    story.append(Paragraph("the eye", s['cover_sub']))
    story.append(Spacer(1, 0.3 * cm))
    _hr(story, GREEN, 2)
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("RETAIL PRESCRIPTION", s['cover_sub']))
    story.append(Spacer(1, 1 * cm))
    
    p_data = [
        ["Patient Name", patient_name, "Date", reviewed_at or datetime.now().strftime("%d %b %Y")],
        ["Age / Gender", f"{patient_age}y / {patient_gender}", "Case ID", case_id]
    ]
    p_table = Table(p_data, colWidths=[3 * cm, 5.5 * cm, 3.5 * cm, 5 * cm])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('BACKGROUND', (2, 0), (2, -1), LIGHT_GRAY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), DARK_BLUE),
        ('TEXTCOLOR', (2, 0), (2, -1), DARK_BLUE),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 1.5 * cm))
    
    _section(story, "Clinical Diagnosis", s)
    story.append(Paragraph(doctor_diagnosis or "Not specified", s['body']))
    story.append(Spacer(1, 0.5 * cm))
    
    _section(story, "Prescription (Rx)", s)
    story.append(Paragraph(doctor_prescription or "No medications prescribed.", s['body']))
    story.append(Spacer(1, 0.5 * cm))
    
    if doctor_referral:
        _section(story, "Referral & Follow Up", s)
        story.append(Paragraph(doctor_referral, s['body']))
        story.append(Spacer(1, 0.5 * cm))
        
    if doctor_notes:
        _section(story, "Additional Notes", s)
        story.append(Paragraph(doctor_notes, s['body']))
        story.append(Spacer(1, 0.5 * cm))
        
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("AI Risk Level Context: " + risk_level, s['note']))
    story.append(Paragraph("Generated by Nayana AI Tele-Ophthalmology. Always consult your doctor before taking any medication.", s['caption']))
    
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return output_path

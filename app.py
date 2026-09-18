"""
แอปติดตามการเติบโตของทารกคลอดก่อนกำหนด — โครงการ "Prachuap Model" 🌸
====================================================================
วัตถุประสงค์: ใช้เป็นเครื่องมือให้กำลังใจคุณแม่หลังคลอดในการเลี้ยงลูกด้วยนมแม่
อย่างต่อเนื่องที่บ้าน "ไม่ใช่" เครื่องมือประเมินพัฒนาการทางการแพทย์แบบสมบูรณ์
(จึงไม่รวมข้อมูลเส้นรอบศีรษะ และแสดงเฉพาะเส้น P50 เพื่อลดความกังวลของคุณแม่)

ระบบระบุตัวตนด้วย "รหัสกลุ่มตัวอย่าง (BF Code)" ที่พยาบาลวิจัยเป็นผู้กำหนดให้
รองรับการเปิดลิงก์ที่แนบรหัสไว้ล่วงหน้า เช่น ?bf=BF001 เพื่อให้คุณแม่ไม่ต้องพิมพ์เอง

หมายเหตุสำคัญด้านข้อมูล: เส้น P50 ในไฟล์นี้เป็น "ข้อมูลจำลอง (mock data)" ที่สร้างขึ้น
ด้วยสูตรทางคณิตศาสตร์เพื่อสาธิตการทำงานของแอปเท่านั้น ไม่ใช่ตาราง Fenton 2025 ฉบับ
ทางการที่ผ่านการรับรองทางคลินิก ก่อนใช้งานจริงกับกลุ่มตัวอย่างควรแทนที่ด้วยชุดข้อมูล
ที่ผ่านการตรวจสอบจากทีมวิจัย/กุมารแพทย์

หมายเหตุด้านการเก็บข้อมูล: แอปนี้พยายามบันทึกประวัติลงไฟล์ JSON ในเครื่องเพื่อจำลอง
การบันทึกถาวรตามรหัส BF Code หากรันบนแพลตฟอร์มที่ไม่อนุญาตให้เขียนไฟล์ (เช่นบาง
บริการ cloud แบบ read-only) แอปจะยังทำงานได้ปกติโดยใช้ st.session_state เก็บข้อมูล
ไว้ชั่วคราวในเซสชันนั้นๆ แทน
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# ตั้งค่าหน้าเพจ + ธีมพาสเทล
# ============================================================================
st.set_page_config(
    page_title="Prachuap Model — ติดตามการเติบโตของหนูน้อย 🌸",
    page_icon="👶",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #FFF9FB 0%, #FFFFFF 35%); }
        div[data-testid="stForm"] {
            background-color: #FFF3F7;
            padding: 1.2rem 1.2rem 0.6rem 1.2rem;
            border-radius: 20px;
            border: 1px solid #FADCE6;
        }
        .big-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.1rem; }
        .sub-title { font-size: 0.95rem; color: #9B8B8F; margin-bottom: 0.8rem; }
        .ga-badge {
            background-color: #E9F5FF; border: 1px solid #CFE8FF; border-radius: 16px;
            padding: 0.7rem 1rem; font-size: 1.0rem; font-weight: 700; color: #3A6EA5;
            text-align: center; margin-bottom: 0.8rem;
        }
        .bf-badge {
            background-color: #F1EBFF; border: 1px solid #DFD2FF; border-radius: 16px;
            padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 700; color: #6A4FA0;
            text-align: center; margin-bottom: 0.8rem;
        }
        div.stButton > button, button[kind="formSubmit"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

WEEK_MIN, WEEK_MAX = 22, 50
DATA_FILE = "bf_growth_data.json"

COLOR_WEIGHT_BOY = "#7FB3E8"
COLOR_WEIGHT_GIRL = "#F2A6C4"
COLOR_LENGTH = "#8FD4A8"
COLOR_P50_LINE = "#B9B9B9"

DISCLAIMER_TEXT = (
    "⚠️ กราฟนี้จัดทำขึ้นเพื่อเป็นกำลังใจในการส่งเสริมการเลี้ยงลูกด้วยนมแม่เท่านั้น "
    "ไม่ได้ใช้ในการประเมินทางการแพทย์แบบสมบูรณ์ หากมีข้อสงสัยเรื่องพัฒนาการ "
    "ควรปรึกษากุมารแพทย์"
)

# ============================================================================
# ข้อมูลจำลองเส้น P50 (Fenton-style) — น้ำหนัก + ความยาว แยกตามเพศ
# ============================================================================
@st.cache_data
def generate_fenton_p50_data():
    """
    สร้างข้อมูลจำลองเส้นค่ากลาง (P50) สไตล์ Fenton แยกตามเพศ (ชาย/หญิง)
    สำหรับน้ำหนัก (กก.) และความยาว (ซม.) อายุครรภ์ปรับแก้ 22-50 สัปดาห์
    (ใช้เฉพาะ P50 ตามข้อกำหนด เพื่อลดความกังวลของคุณแม่)
    """
    weeks = np.arange(WEEK_MIN, WEEK_MAX + 1)

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    weight_params = {
        "boy":  dict(L=6.1, k=0.145, x0=35.5, base=0.30),
        "girl": dict(L=5.7, k=0.148, x0=35.0, base=0.28),
    }
    length_params = {
        "boy":  dict(L=34.0, k=0.135, x0=34.5, base=24.5),
        "girl": dict(L=33.0, k=0.138, x0=34.0, base=24.0),
    }

    data = {}
    for sex in ["boy", "girl"]:
        wp = weight_params[sex]
        weight_p50 = wp["base"] + logistic(weeks, wp["L"], wp["k"], wp["x0"])
        lp = length_params[sex]
        length_p50 = lp["base"] + logistic(weeks, lp["L"], lp["k"], lp["x0"])
        data[sex] = {
            "weight": pd.DataFrame({"week": weeks, "P50": weight_p50}),
            "length": pd.DataFrame({"week": weeks, "P50": length_p50}),
        }
    return data


FENTON_P50 = generate_fenton_p50_data()


def get_p50_weight_at_ga(sex: str, ga_weeks: float) -> float:
    """คืนค่าน้ำหนัก P50 (กก.) โดยประมาณค่า ณ อายุครรภ์ปรับแก้ที่กำหนด (interpolation)"""
    df = FENTON_P50[sex]["weight"]
    ga_clamped = min(max(ga_weeks, WEEK_MIN), WEEK_MAX)
    return float(np.interp(ga_clamped, df["week"], df["P50"]))


# ============================================================================
# การจัดเก็บ/โหลดข้อมูลตามรหัส BF Code — จำลองด้วยไฟล์ JSON ในเครื่อง
# ============================================================================
def load_all_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_all_data(data: dict) -> bool:
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        # เขียนไฟล์ไม่ได้ (เช่นระบบ read-only) — ยังทำงานต่อได้ด้วย session_state
        return False


if "all_data" not in st.session_state:
    st.session_state.all_data = load_all_data()

# ============================================================================
# ข้อความให้กำลังใจ / คำแนะนำ
# ============================================================================
ABOVE_P50_MESSAGES = [
    "เก่งมากเลยค่ะคุณแม่! 💖 น้ำหนักและส่วนสูงของหนูเติบโตตามเกณฑ์เลย พลังน้ำนมแม่วิเศษที่สุด สู้ต่อไปนะคะ!",
    "สุดยอดไปเลยค่ะ! 🌟 หนูน้อยโตดีมาก คุณแม่คือฮีโร่ตัวจริงของลูกเลยค่ะ",
    "ยอดเยี่ยมมากค่ะ! 🍼 ทุกหยดน้ำนมของคุณแม่คือพลังที่ทำให้หนูแข็งแรงขึ้นทุกวัน",
]
BELOW_P50_MESSAGES = [
    "หนูเติบโตตามจังหวะของตัวเองค่ะ ช่วงนี้น้ำหนักอาจจะยังไม่ถึงเส้นเกณฑ์เป๊ะๆ คุณแม่ไม่ต้องกังวลนะคะ "
    "แนะนำให้คุณแม่ให้นมอย่างสม่ำเสมอ และนำข้อมูลนี้ไปปรึกษาคุณหมอหรือพยาบาลนมแม่ในวันนัด "
    "เพื่อรับคำแนะนำดีๆ เพิ่มเติมค่ะ ✌️",
    "ไม่เป็นไรเลยค่ะ ทารกแต่ละคนมีจังหวะการเติบโตไม่เหมือนกัน 🤍 ให้นมแม่ต่อเนื่องนะคะ "
    "และลองพูดคุยกับพยาบาลนมแม่หรือกุมารแพทย์ในนัดครั้งถัดไปเพื่อความสบายใจค่ะ",
]
FIRST_ENTRY_MESSAGES = [
    "ยินดีต้อนรับสู่เส้นทางการเติบโตของหนูน้อยนะคะ 💖 คุณแม่เก่งมากที่ใส่ใจดูแลขนาดนี้ค่ะ",
]

# ============================================================================
# BF Code — รับค่าจาก query_params หรือให้คุณแม่กรอกเอง
# ============================================================================
query_params = st.query_params
default_bf_code = query_params.get("bf", "")

st.markdown('<div class="big-title">🌸 Prachuap Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">แอปติดตามการเติบโตของหนูน้อย เพื่อเป็นกำลังใจให้คุณแม่เลี้ยงลูกด้วยนมแม่</div>',
    unsafe_allow_html=True,
)
st.warning(DISCLAIMER_TEXT)

with st.sidebar:
    st.markdown("### 🔑 ระบุตัวตน")
    bf_code = st.text_input(
        "รหัสกลุ่มตัวอย่าง (BF Code)",
        value=default_bf_code,
        help="รหัสนี้ได้รับจากพยาบาลวิจัยของโครงการ Prachuap Model กรุณากรอกให้ตรงกันทุกครั้ง "
             "เพื่อป้องกันข้อมูลสลับกับคุณแม่ท่านอื่น",
        placeholder="เช่น BF001",
    ).strip().upper()

    if default_bf_code and bf_code == default_bf_code.strip().upper():
        st.caption(f"✅ ดึงรหัส **{default_bf_code}** จากลิงก์ที่พยาบาลส่งให้อัตโนมัติ")

    st.divider()

if not bf_code:
    st.info("👈 กรุณากรอก **รหัสกลุ่มตัวอย่าง (BF Code)** ในแถบด้านซ้ายก่อนเริ่มบันทึกข้อมูลนะคะ")
    st.stop()

st.markdown(f'<div class="bf-badge">🔑 กำลังบันทึกข้อมูลของรหัส: {bf_code}</div>', unsafe_allow_html=True)

if bf_code not in st.session_state.all_data:
    st.session_state.all_data[bf_code] = []

# ============================================================================
# Sidebar: ฟอร์มกรอกข้อมูล
# ============================================================================
with st.sidebar:
    st.markdown("### 📝 กรอกข้อมูลวันนี้")
    sex_label = st.radio("👶 เพศของทารก", ["ชาย", "หญิง"], horizontal=True)

    st.markdown("**📅 ข้อมูลสำหรับคำนวณอายุครรภ์ปรับแก้**")
    birth_date = st.date_input(
        "วันเกิดของหนูน้อย", value=date(date.today().year, date.today().month, 1),
        max_value=date.today(),
    )
    col_w, col_d = st.columns(2)
    with col_w:
        ga_birth_weeks = st.number_input("อายุครรภ์แรกเกิด (สัปดาห์)", min_value=22, max_value=42, value=32, step=1)
    with col_d:
        ga_birth_days = st.number_input("+ วัน", min_value=0, max_value=6, value=0, step=1)

    entry_date = st.date_input("วันที่บันทึกข้อมูลวันนี้", value=date.today())

    # ---- คำนวณอายุครรภ์ปรับแก้ (Corrected GA) อัตโนมัติ ----
    chronological_age_days = (entry_date - birth_date).days
    total_birth_ga_days = (ga_birth_weeks * 7) + ga_birth_days
    corrected_ga_days = max(total_birth_ga_days + chronological_age_days, 0)
    corrected_ga_weeks_int = corrected_ga_days // 7
    corrected_ga_remainder_days = corrected_ga_days % 7
    corrected_ga_weeks_decimal = corrected_ga_days / 7.0

    st.markdown(
        f'<div class="ga-badge">📐 อายุครรภ์ปรับแก้ปัจจุบัน:<br>'
        f'{corrected_ga_weeks_int} สัปดาห์ {corrected_ga_remainder_days} วัน</div>',
        unsafe_allow_html=True,
    )

    with st.form("input_form", clear_on_submit=False):
        st.markdown("**⚖️ ข้อมูลการเจริญเติบโต**")
        current_weight = st.number_input(
            "น้ำหนักปัจจุบัน (กก.)", min_value=0.2, max_value=10.0, value=1.8, step=0.01, format="%.2f",
        )
        current_length = st.number_input(
            "ความยาวปัจจุบัน (ซม.)", min_value=20.0, max_value=70.0, value=42.0, step=0.1,
        )
        submitted = st.form_submit_button("💾 บันทึกข้อมูล", use_container_width=True)

    if st.session_state.all_data.get(bf_code):
        st.write("")
        if st.button("🗑️ ล้างข้อมูลของรหัสนี้", use_container_width=True):
            st.session_state.all_data[bf_code] = []
            save_all_data(st.session_state.all_data)
            st.rerun()

sex_key = "boy" if sex_label == "ชาย" else "girl"

# ============================================================================
# บันทึกข้อมูล + ข้อความให้กำลังใจ / คำแนะนำ (ตามเงื่อนไข P50)
# ============================================================================
if submitted:
    new_record = {
        "sex": sex_key,
        "entry_date": entry_date.isoformat(),
        "corrected_age_weeks": round(corrected_ga_weeks_decimal, 2),
        "corrected_age_label": f"{corrected_ga_weeks_int} สัปดาห์ {corrected_ga_remainder_days} วัน",
        "weight": current_weight,
        "length": current_length,
    }
    is_first_entry = len(st.session_state.all_data[bf_code]) == 0
    st.session_state.all_data[bf_code].append(new_record)
    save_all_data(st.session_state.all_data)

    if is_first_entry:
        st.success(np.random.choice(FIRST_ENTRY_MESSAGES))
    else:
        p50_weight_now = get_p50_weight_at_ga(sex_key, corrected_ga_weeks_decimal)
        if current_weight >= p50_weight_now:
            st.success(np.random.choice(ABOVE_P50_MESSAGES))
            st.balloons()
        else:
            st.warning(np.random.choice(BELOW_P50_MESSAGES))

# ============================================================================
# หน้าหลัก — กราฟรวม (Dual Y-Axis) เฉพาะเส้น P50
# ============================================================================
records = st.session_state.all_data.get(bf_code, [])
records_df = pd.DataFrame(records).sort_values("corrected_age_weeks") if records else pd.DataFrame()

weight_color = COLOR_WEIGHT_BOY if sex_key == "boy" else COLOR_WEIGHT_GIRL
baby_emoji = "👦" if sex_key == "boy" else "👧"
weight_p50_df = FENTON_P50[sex_key]["weight"]
length_p50_df = FENTON_P50[sex_key]["length"]

fig = make_subplots(specs=[[{"secondary_y": True}]])

# --- เส้น P50 น้ำหนัก (แกนซ้าย) — "เส้นตามเกณฑ์" จางๆ ---
fig.add_trace(
    go.Scatter(
        x=weight_p50_df["week"], y=weight_p50_df["P50"],
        mode="lines", name="น้ำหนัก: เส้นตามเกณฑ์",
        line=dict(color=COLOR_P50_LINE, width=2.0, dash="dash"),
        opacity=0.8,
        hovertemplate="น้ำหนักตามเกณฑ์: %{y:.2f} กก.<extra></extra>",
    ),
    secondary_y=False,
)

# --- เส้น P50 ความยาว (แกนขวา) — "เส้นตามเกณฑ์" จางๆ ---
fig.add_trace(
    go.Scatter(
        x=length_p50_df["week"], y=length_p50_df["P50"],
        mode="lines", name="ความยาว: เส้นตามเกณฑ์",
        line=dict(color=COLOR_P50_LINE, width=2.0, dash="dot"),
        opacity=0.6,
        hovertemplate="ความยาวตามเกณฑ์: %{y:.2f} ซม.<extra></extra>",
    ),
    secondary_y=True,
)

# --- เส้นข้อมูลของหนูน้อย ---
if not records_df.empty:
    fig.add_trace(
        go.Scatter(
            x=records_df["corrected_age_weeks"], y=records_df["weight"],
            mode="lines+text", name=f"น้ำหนักของหนู {baby_emoji}",
            line=dict(color=weight_color, width=5),
            text=[baby_emoji] * len(records_df),
            textposition="middle center", textfont=dict(size=20),
            hovertemplate="น้ำหนักของหนู: %{y:.2f} กก.<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=records_df["corrected_age_weeks"], y=records_df["length"],
            mode="lines+text", name=f"ความยาวของหนู {baby_emoji}",
            line=dict(color=COLOR_LENGTH, width=5),
            text=[baby_emoji] * len(records_df),
            textposition="middle center", textfont=dict(size=20),
            hovertemplate="ความยาวของหนู: %{y:.2f} ซม.<extra></extra>",
        ),
        secondary_y=True,
    )

fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5, font=dict(size=11)),
    margin=dict(t=90, b=10, l=10, r=10),
    height=500,
    font=dict(size=13, color="#5A5A5A"),
)
fig.update_xaxes(
    title="อายุครรภ์ปรับแก้ (สัปดาห์)", range=[WEEK_MIN, WEEK_MAX], dtick=4,
    showgrid=False, showline=True, linecolor="#E5E5E5", zeroline=False,
)
fig.update_yaxes(
    title_text="น้ำหนัก (กก.)", secondary_y=False,
    showgrid=True, gridcolor="#F2F2F2", gridwidth=1, showline=False, zeroline=False,
    title_font=dict(color=weight_color), tickfont=dict(color=weight_color),
)
fig.update_yaxes(
    title_text="ความยาว (ซม.)", secondary_y=True,
    showgrid=False, showline=False, zeroline=False,
    title_font=dict(color=COLOR_LENGTH), tickfont=dict(color=COLOR_LENGTH),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# ประวัติข้อมูล
# ----------------------------------------------------------------------------
with st.expander("📖 ดูประวัติการบันทึกข้อมูลของรหัสนี้"):
    if not records_df.empty:
        display_df = records_df.rename(columns={
            "entry_date": "วันที่บันทึก",
            "corrected_age_label": "อายุครรภ์ปรับแก้",
            "weight": "น้ำหนัก (กก.)",
            "length": "ความยาว (ซม.)",
        })[["วันที่บันทึก", "อายุครรภ์ปรับแก้", "น้ำหนัก (กก.)", "ความยาว (ซม.)"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.write("ยังไม่มีข้อมูลของรหัสนี้ — กรอกข้อมูลในแถบด้านซ้ายเพื่อเริ่มติดตามค่ะ 💛")

st.divider()
st.caption(DISCLAIMER_TEXT)
st.caption("🔒 ข้อมูลนี้ใช้เพื่อโครงการวิจัย Prachuap Model เท่านั้น กรุณาเก็บรหัส BF Code ของท่านไว้เป็นความลับ")

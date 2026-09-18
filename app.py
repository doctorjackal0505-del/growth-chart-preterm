"""
แอปติดตามการเติบโตของทารกคลอดก่อนกำหนด (Preterm Growth Tracker) 🌸
====================================================================
ออกแบบมาให้คุณแม่หลังคลอดที่อาจเหนื่อยล้าใช้งานได้ง่าย น่ารัก อบอุ่นใจ
- คำนวณ "อายุครรภ์ปรับแก้ (Corrected GA)" ให้อัตโนมัติจากวันเกิด + อายุครรภ์แรกเกิด
- ติดตามน้ำหนัก (กก.) และความยาว (ซม.) ในกราฟเดียว แบบ dual Y-axis
- ให้กำลังใจการเลี้ยงลูกด้วยนมแม่อย่างต่อเนื่อง

หมายเหตุสำคัญ: เส้นเปอร์เซ็นไทล์ (P3/P50/P97) ในไฟล์นี้เป็น "ข้อมูลจำลอง (mock data)"
ที่สร้างขึ้นด้วยสูตรทางคณิตศาสตร์เพื่อสาธิตการทำงานของแอปเท่านั้น ไม่ใช่ตาราง
Fenton 2025 ฉบับทางการที่ผ่านการรับรองทางคลินิก ก่อนใช้งานจริงกับผู้ป่วยควรแทนที่
ด้วยชุดข้อมูลที่ผ่านการตรวจสอบ และใช้ร่วมกับดุลยพินิจของกุมารแพทย์เสมอ
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, datetime

# ============================================================================
# ตั้งค่าหน้าเพจ + ธีมน่ารัก พาสเทล
# ============================================================================
st.set_page_config(
    page_title="ติดตามการเติบโตของหนูน้อย 🌸",
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
        .big-title { font-size: 1.7rem; font-weight: 800; margin-bottom: 0.1rem; }
        .sub-title { font-size: 1.0rem; color: #9B8B8F; margin-bottom: 1.0rem; }
        .ga-badge {
            background-color: #E9F5FF;
            border: 1px solid #CFE8FF;
            border-radius: 16px;
            padding: 0.7rem 1rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #3A6EA5;
            text-align: center;
            margin-bottom: 0.8rem;
        }
        div.stButton > button, button[kind="formSubmit"] {
            border-radius: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

WEEK_MIN, WEEK_MAX = 22, 50
Z_SCORES = {3: -1.881, 50: 0.0, 97: 1.881}

COLOR_WEIGHT_BOY = "#7FB3E8"    # ฟ้าพาสเทล
COLOR_WEIGHT_GIRL = "#F2A6C4"   # ชมพูพาสเทล
COLOR_LENGTH = "#8FD4A8"        # เขียวมิ้นต์พาสเทล
COLOR_REF_LINE = "#C9C9C9"      # เทาอ่อนจางๆ สำหรับเส้นอ้างอิง


# ============================================================================
# ข้อมูลจำลองเส้น Fenton 2025 (น้ำหนัก + ความยาว) แยกตามเพศ — hardcoded ผ่านสูตร
# ============================================================================
@st.cache_data
def generate_fenton_mock_data():
    """
    สร้างข้อมูลจำลองเส้นโค้งการเติบโตสไตล์ Fenton แยกตามเพศ (ชาย/หญิง)
    สำหรับน้ำหนัก (กก.) และความยาว (ซม.) อายุครรภ์ปรับแก้ 22-50 สัปดาห์
    เฉพาะเปอร์เซ็นไทล์ P3 / P50 / P97 (ใช้ logistic growth curve + z-score)
    """
    weeks = np.arange(WEEK_MIN, WEEK_MAX + 1)

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    data = {"boy": {}, "girl": {}}

    weight_params = {
        "boy":  dict(L=6.1, k=0.145, x0=35.5, base=0.30),
        "girl": dict(L=5.7, k=0.148, x0=35.0, base=0.28),
    }
    length_params = {
        "boy":  dict(L=34.0, k=0.135, x0=34.5, base=24.5),
        "girl": dict(L=33.0, k=0.138, x0=34.0, base=24.0),
    }

    for sex in ["boy", "girl"]:
        wp = weight_params[sex]
        w_median = wp["base"] + logistic(weeks, wp["L"], wp["k"], wp["x0"])
        w_sd = 0.13 * w_median + 0.02
        weight_df = pd.DataFrame({
            "week": weeks,
            "P3": w_median + Z_SCORES[3] * w_sd,
            "P50": w_median,
            "P97": w_median + Z_SCORES[97] * w_sd,
        })

        lp = length_params[sex]
        l_median = lp["base"] + logistic(weeks, lp["L"], lp["k"], lp["x0"])
        l_sd = 0.045 * l_median + 0.3
        length_df = pd.DataFrame({
            "week": weeks,
            "P3": l_median + Z_SCORES[3] * l_sd,
            "P50": l_median,
            "P97": l_median + Z_SCORES[97] * l_sd,
        })

        data[sex]["weight"] = weight_df
        data[sex]["length"] = length_df

    return data


FENTON_DATA = generate_fenton_mock_data()

# ============================================================================
# session_state — เก็บประวัติเป็น Pandas DataFrame
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "sex", "entry_date", "corrected_age_weeks", "corrected_age_label", "weight", "length"
    ])

# ============================================================================
# ข้อความให้กำลังใจ
# ============================================================================
WEIGHT_UP_MESSAGES = [
    "เก่งมากเลยค่ะคุณแม่! 💖 น้ำหนักหนูขึ้นแล้ว น้ำนมแม่คือยารักษาและภูมิคุ้มกันที่ดีที่สุดเลย ทำต่อไปนะคะ!",
    "สุดยอดไปเลยค่ะ! 🌟 น้ำหนักน้องขึ้นสวยมาก คุณแม่คือฮีโร่ตัวจริงของหนูน้อยเลยค่ะ",
    "ยอดเยี่ยมมากค่ะ! 🍼 ทุกหยดน้ำนมของคุณแม่คือพลังที่ทำให้หนูแข็งแรงขึ้นทุกวัน สู้ๆ นะคะ",
    "เยี่ยมไปเลยค่ะคุณแม่! ✨ การเติบโตของหนูวันนี้คือผลลัพธ์จากความรักและความพยายามของคุณแม่ล้วนๆ",
]
STEADY_MESSAGES = [
    "ไม่เป็นไรเลยค่ะ ทุกก้าวเล็กๆ ของหนูก็คือความสำเร็จของคุณแม่ 🤍 อย่าเพิ่งกังวลไปนะคะ",
    "การเติบโตของทารกคลอดก่อนกำหนดต้องใช้เวลาค่ะ คุณแม่ทำดีที่สุดแล้ว 🌼 ให้กำลังใจตัวเองด้วยนะคะ",
]
FIRST_ENTRY_MESSAGES = [
    "ยินดีต้อนรับสู่เส้นทางการเติบโตของหนูน้อยนะคะ 💖 คุณแม่เก่งมากที่ใส่ใจดูแลขนาดนี้ค่ะ",
]

# ============================================================================
# Sidebar: ฟอร์มกรอกข้อมูล
# ============================================================================
with st.sidebar:
    st.markdown("### 🌸 กรอกข้อมูลวันนี้")
    st.caption("กรอกทุกครั้งที่ชั่งน้ำหนัก/วัดตัวหนูน้อยนะคะ")

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
    corrected_ga_days = total_birth_ga_days + chronological_age_days
    corrected_ga_days = max(corrected_ga_days, 0)
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

    if not st.session_state.history.empty:
        st.write("")
        if st.button("🗑️ ล้างข้อมูลทั้งหมด", use_container_width=True):
            st.session_state.history = st.session_state.history.iloc[0:0]
            st.rerun()

sex_key = "boy" if sex_label == "ชาย" else "girl"

# ============================================================================
# บันทึกข้อมูล + ข้อความให้กำลังใจ
# ============================================================================
if submitted:
    new_row = {
        "sex": sex_key,
        "entry_date": entry_date,
        "corrected_age_weeks": round(corrected_ga_weeks_decimal, 2),
        "corrected_age_label": f"{corrected_ga_weeks_int} สัปดาห์ {corrected_ga_remainder_days} วัน",
        "weight": current_weight,
        "length": current_length,
    }
    prev_rows = st.session_state.history[st.session_state.history["sex"] == sex_key]

    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])], ignore_index=True
    )

    if prev_rows.empty:
        st.success(np.random.choice(FIRST_ENTRY_MESSAGES))
    else:
        last_row = prev_rows.sort_values("corrected_age_weeks").iloc[-1]
        if current_weight > last_row["weight"]:
            st.success(np.random.choice(WEIGHT_UP_MESSAGES))
            st.balloons()
        else:
            st.info(np.random.choice(STEADY_MESSAGES))

# ============================================================================
# หน้าหลัก
# ============================================================================
st.markdown('<div class="big-title">🌸 การเติบโตของหนูน้อย</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">ติดตามน้ำหนัก 👶 และความยาว 🍼 เทียบกับกราฟมาตรฐานสำหรับทารกคลอดก่อนกำหนด</div>',
    unsafe_allow_html=True,
)

baby_records = st.session_state.history[st.session_state.history["sex"] == sex_key].sort_values("corrected_age_weeks")
weight_color = COLOR_WEIGHT_BOY if sex_key == "boy" else COLOR_WEIGHT_GIRL
weight_df = FENTON_DATA[sex_key]["weight"]
length_df = FENTON_DATA[sex_key]["length"]

# ----------------------------------------------------------------------------
# กราฟรวม: Dual Y-Axis (น้ำหนัก ซ้าย / ความยาว ขวา)
# ----------------------------------------------------------------------------
fig = make_subplots(specs=[[{"secondary_y": True}]])

# --- โซนแรเงา "พลังนมแม่ล้วน" (สัปดาห์ 22-40) ---
fig.add_vrect(
    x0=WEEK_MIN, x1=40,
    fillcolor="#FFE3EE" if sex_key == "girl" else "#E3F0FF",
    opacity=0.45, line_width=0, layer="below",
)
fig.add_annotation(
    x=WEEK_MIN + 0.4, y=1.0, xref="x", yref="paper",
    text="✨ พลังนมแม่ล้วน",
    showarrow=False,
    font=dict(size=13, color="#B15C86" if sex_key == "girl" else "#3A6EA5"),
    align="left", xanchor="left", yanchor="top",
)

# --- เส้นอ้างอิงน้ำหนัก P3/P50/P97 (แกนซ้าย) — จางๆ เส้นประ ---
for pct, dash in [(97, "dash"), (50, "solid"), (3, "dash")]:
    fig.add_trace(
        go.Scatter(
            x=weight_df["week"], y=weight_df[f"P{pct}"],
            mode="lines", name=f"น้ำหนัก P{pct} ({'เกณฑ์สูง' if pct==97 else 'ตามเกณฑ์' if pct==50 else 'เกณฑ์ต่ำ'})",
            line=dict(color=COLOR_REF_LINE, width=1.4 if pct != 50 else 1.8, dash=dash),
            opacity=0.75,
            hovertemplate=f"น้ำหนัก P{pct}: %{{y:.2f}} กก.<extra></extra>",
        ),
        secondary_y=False,
    )

# --- เส้นอ้างอิงความยาว P3/P50/P97 (แกนขวา) — จางๆ เส้นประ ---
for pct, dash in [(97, "dash"), (50, "solid"), (3, "dash")]:
    fig.add_trace(
        go.Scatter(
            x=length_df["week"], y=length_df[f"P{pct}"],
            mode="lines", name=f"ความยาว P{pct} ({'เกณฑ์สูง' if pct==97 else 'ตามเกณฑ์' if pct==50 else 'เกณฑ์ต่ำ'})",
            line=dict(color=COLOR_REF_LINE, width=1.4 if pct != 50 else 1.8, dash=dash),
            opacity=0.55,
            hovertemplate=f"ความยาว P{pct}: %{{y:.2f}} ซม.<extra></extra>",
        ),
        secondary_y=True,
    )

# --- เส้นน้ำหนักของหนูน้อย (แกนซ้าย) — หนา พาสเทล + emoji 👶 ---
if not baby_records.empty:
    fig.add_trace(
        go.Scatter(
            x=baby_records["corrected_age_weeks"], y=baby_records["weight"],
            mode="lines+text", name="น้ำหนักของหนู 👶",
            line=dict(color=weight_color, width=5),
            text=["👶"] * len(baby_records),
            textposition="middle center", textfont=dict(size=20),
            hovertemplate="น้ำหนักของหนู: %{y:.2f} กก.<extra></extra>",
        ),
        secondary_y=False,
    )
    # --- เส้นความยาวของหนูน้อย (แกนขวา) — หนา เขียวมิ้นต์ + emoji 🍼 ---
    fig.add_trace(
        go.Scatter(
            x=baby_records["corrected_age_weeks"], y=baby_records["length"],
            mode="lines+text", name="ความยาวของหนู 🍼",
            line=dict(color=COLOR_LENGTH, width=5),
            text=["🍼"] * len(baby_records),
            textposition="middle center", textfont=dict(size=20),
            hovertemplate="ความยาวของหนู: %{y:.2f} ซม.<extra></extra>",
        ),
        secondary_y=True,
    )

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.16, xanchor="center", x=0.5,
        font=dict(size=10),
    ),
    margin=dict(t=110, b=10, l=10, r=10),
    height=520,
    font=dict(size=13, color="#5A5A5A"),
)
fig.update_xaxes(
    title="อายุครรภ์ปรับแก้ (สัปดาห์)",
    range=[WEEK_MIN, WEEK_MAX], dtick=4,
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
with st.expander("📖 ดูประวัติการบันทึกข้อมูล"):
    if not baby_records.empty:
        display_df = baby_records.rename(columns={
            "entry_date": "วันที่บันทึก",
            "corrected_age_label": "อายุครรภ์ปรับแก้",
            "weight": "น้ำหนัก (กก.)",
            "length": "ความยาว (ซม.)",
        })[["วันที่บันทึก", "อายุครรภ์ปรับแก้", "น้ำหนัก (กก.)", "ความยาว (ซม.)"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.write("ยังไม่มีข้อมูล — กรอกข้อมูลในแถบด้านซ้ายเพื่อเริ่มติดตามค่ะ 💛")

st.divider()
st.caption(
    "⚠️ เส้นเปอร์เซ็นไทล์ในแอปนี้เป็นข้อมูลจำลองเพื่อสาธิตการทำงานเท่านั้น ไม่ใช่ตาราง "
    "Fenton 2025 ฉบับทางการ กรุณาใช้ร่วมกับคำแนะนำของกุมารแพทย์หรือบุคลากรทางการแพทย์เสมอ 💛"
)

"""
แอปติดตามการเติบโตของทารกคลอดก่อนกำหนด (Preterm Growth Tracker)
====================================================================
ออกแบบสำหรับคุณแม่หลังคลอดที่อาจเหนื่อยล้าและพักผ่อนไม่เพียงพอ
เน้นความเรียบง่าย อ่านง่าย ไม่ซับซ้อน และให้กำลังใจในการให้นมแม่

- ติดตามเฉพาะ "น้ำหนัก" และ "ความยาว" เท่านั้น (ไม่มีเส้นรอบศีรษะ)
- กราฟทั้งสองแยกกันชัดเจน ไม่ใช้แกน Y คู่ (dual-axis) เพื่อไม่ให้สับสน
- กราฟน้ำหนักอยู่บนสุด เพราะสะท้อนผลของการให้นมแม่ได้ไวที่สุด

หมายเหตุสำคัญ: เส้นเปอร์เซ็นไทล์ในไฟล์นี้เป็น "ข้อมูลจำลอง (mock data)" ที่สร้างขึ้น
เพื่อสาธิตการทำงานของแอปเท่านั้น ไม่ใช่ตาราง Fenton 2025 ฉบับทางการที่ผ่านการรับรอง
ก่อนใช้งานจริงกับผู้ป่วยควรแทนที่ด้วยชุดข้อมูลที่ผ่านการตรวจสอบ และใช้ร่วมกับ
ดุลยพินิจของกุมารแพทย์เสมอ
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ============================================================================
# ตั้งค่าหน้าเพจ + ธีมสีอ่อนโยน สบายตา
# ============================================================================
st.set_page_config(
    page_title="ติดตามการเติบโตของน้อง",
    page_icon="🌷",
    layout="centered",   # centered = อ่านง่ายบนมือถือ ไม่ต้องเลื่อนซ้ายขวา
)

# สีโทนอ่อน สบายตา ไม่ฉูดฉาด เหมาะกับคุณแม่ที่เหนื่อยล้า
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#4A4A4A"
COLOR_MOTHER_LINE = "#3E7C59"      # เขียวสงบ หนักแน่น อบอุ่น
COLOR_MOTHER_MARKER = "#2E5F45"
COLOR_P50 = "#8FB8DE"              # ฟ้าพาสเทล (เส้นค่ากลาง)
COLOR_P_OUTER = "#F3C9C9"          # ชมพูอ่อนมาก (3rd/97th)
COLOR_P_MID = "#F6DDB0"            # ส้มพีชอ่อน (10th/90th)
COLOR_GRID = "#EDEDED"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLOR_BG}; }}
        h1, h2, h3, p, label, span, div {{ color: {COLOR_TEXT}; }}
        div[data-testid="stForm"] {{
            background-color: #FAFAF7;
            padding: 1.2rem 1.2rem 0.4rem 1.2rem;
            border-radius: 16px;
        }}
        .big-title {{ font-size: 1.65rem; font-weight: 700; margin-bottom: 0.2rem; }}
        .sub-title {{ font-size: 1.0rem; color: #7A7A7A; margin-bottom: 1.2rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PERCENTILES = [3, 10, 50, 90, 97]
Z_SCORES = {3: -1.881, 10: -1.282, 50: 0.0, 90: 1.282, 97: 1.881}
WEEK_MIN, WEEK_MAX = 22, 50

PERCENTILE_STYLE = {
    3:  dict(color=COLOR_P_OUTER, width=1.2, dash="dot",  label="P3"),
    10: dict(color=COLOR_P_MID,   width=1.4, dash="dot",  label="P10"),
    50: dict(color=COLOR_P50,     width=2.4, dash="solid", label="P50 (ค่ากลาง)"),
    90: dict(color=COLOR_P_MID,   width=1.4, dash="dot",  label="P90"),
    97: dict(color=COLOR_P_OUTER, width=1.2, dash="dot",  label="P97"),
}


# ============================================================================
# ข้อมูลจำลองเส้นเปอร์เซ็นไทล์ (Fenton-style mock curves) — hardcoded ผ่านสูตร
# ============================================================================
@st.cache_data
def generate_fenton_mock_data():
    """
    สร้างข้อมูลจำลองเส้นโค้งการเติบโตสไตล์ Fenton แยกตามเพศ (ชาย/หญิง)
    สำหรับน้ำหนัก (กก.) และความยาว (ซม.) อายุครรภ์ 22-50 สัปดาห์
    ใช้เส้นโค้งลอจิสติก (logistic growth) + z-score ของแต่ละเปอร์เซ็นไทล์
    เพื่อให้ได้รูปทรงใกล้เคียงกราฟ Fenton จริง
    """
    weeks = np.arange(WEEK_MIN, WEEK_MAX + 1)

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    data = {"boy": {}, "girl": {}}

    # ---------------- น้ำหนัก (กก.) ----------------
    weight_params = {
        "boy":  dict(L=6.1, k=0.145, x0=35.5, base=0.30),
        "girl": dict(L=5.7, k=0.148, x0=35.0, base=0.28),
    }
    for sex, p in weight_params.items():
        median = p["base"] + logistic(weeks, p["L"], p["k"], p["x0"])
        sd = 0.13 * median + 0.02
        df = pd.DataFrame({"week": weeks, "P50": median})
        for pct in PERCENTILES:
            if pct != 50:
                df[f"P{pct}"] = median + Z_SCORES[pct] * sd
        data[sex]["weight"] = df[["week"] + [f"P{p_}" for p_ in PERCENTILES]]

    # ---------------- ความยาว (ซม.) ----------------
    length_params = {
        "boy":  dict(L=34.0, k=0.135, x0=34.5, base=24.5),
        "girl": dict(L=33.0, k=0.138, x0=34.0, base=24.0),
    }
    for sex, p in length_params.items():
        median = p["base"] + logistic(weeks, p["L"], p["k"], p["x0"])
        sd = 0.045 * median + 0.3
        df = pd.DataFrame({"week": weeks, "P50": median})
        for pct in PERCENTILES:
            if pct != 50:
                df[f"P{pct}"] = median + Z_SCORES[pct] * sd
        data[sex]["length"] = df[["week"] + [f"P{p_}" for p_ in PERCENTILES]]

    return data


FENTON_DATA = generate_fenton_mock_data()

# ============================================================================
# session_state
# ============================================================================
if "records" not in st.session_state:
    st.session_state.records = []


# ============================================================================
# ฟังก์ชันวาดกราฟ — เรียบง่าย ไม่มีเส้นตารางหนัก ไม่มีแกน Y คู่
# ============================================================================
def plot_growth_chart(measure_key: str, sex: str, unit: str, chart_title: str,
                       records_df: pd.DataFrame, accent_emoji: str):
    df = FENTON_DATA[sex][measure_key]
    fig = go.Figure()

    # เส้นเปอร์เซ็นไทล์พื้นหลัง — บางและจาง ไม่แย่งความสนใจ
    for pct in PERCENTILES:
        style = PERCENTILE_STYLE[pct]
        fig.add_trace(
            go.Scatter(
                x=df["week"], y=df[f"P{pct}"],
                mode="lines",
                name=style["label"],
                line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                hovertemplate=f"{style['label']}: %{{y:.2f}} {unit}<extra></extra>",
            )
        )

    # เส้นข้อมูลจริงของน้อง — หนา เด่น มี marker ใหญ่ มองเห็นทันที
    if records_df is not None and not records_df.empty:
        fig.add_trace(
            go.Scatter(
                x=records_df["corrected_age"],
                y=records_df[measure_key],
                mode="lines+markers",
                name="น้องของคุณแม่",
                line=dict(color=COLOR_MOTHER_LINE, width=5),
                marker=dict(size=14, color=COLOR_MOTHER_MARKER,
                            line=dict(width=2, color="white")),
                hovertemplate=f"น้อง: %{{y:.2f}} {unit}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=f"{accent_emoji}  {chart_title}", font=dict(size=20)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=60, b=10, l=10, r=10),
        height=380,
        font=dict(size=13, color=COLOR_TEXT),
    )
    # แกน X: เส้นบางเดียว ไม่มีตารางถี่รก
    fig.update_xaxes(
        title="อายุครรภ์ (สัปดาห์)",
        range=[WEEK_MIN, WEEK_MAX],
        dtick=4,
        showgrid=False,
        showline=True,
        linecolor=COLOR_GRID,
        zeroline=False,
    )
    # แกน Y: เส้นตารางแนวนอนบางๆ เท่านั้น พอให้กะระดับได้ ไม่รก
    fig.update_yaxes(
        title=f"{unit}",
        showgrid=True,
        gridcolor=COLOR_GRID,
        gridwidth=1,
        showline=False,
        zeroline=False,
    )
    return fig


# ============================================================================
# ข้อความให้กำลังใจ (สุ่มเพื่อไม่ให้ซ้ำจำเจ)
# ============================================================================
WEIGHT_UP_MESSAGES = [
    "ยอดเยี่ยมมากคุณแม่! น้ำหนักลูกขึ้นแล้ว น้ำนมแม่มีค่าที่สุดเลยค่ะ 💛",
    "คุณแม่เก่งมากๆ ที่พยายามเพื่อลูกขนาดนี้ น้ำหนักน้องขึ้นสวยเลยค่ะ 🌟",
    "เก่งมากค่ะ! ทุกหยดน้ำนมของแม่คือพลังที่ทำให้น้องแข็งแรงขึ้นทุกวัน 🍼",
    "สุดยอดไปเลยค่ะ น้ำหนักที่เพิ่มขึ้นคือหลักฐานความรักและความพยายามของคุณแม่ 🌸",
]
LENGTH_UP_MESSAGES = [
    "น้องตัวยาวขึ้นแล้วนะคะ! นมแม่ช่วยให้น้องเติบโตได้เต็มศักยภาพเลยค่ะ 🌷",
    "เก่งมากค่ะ ความยาวตัวของน้องเพิ่มขึ้น แปลว่าโภชนาการกำลังไปได้สวย 💪",
]
STEADY_MESSAGES = [
    "ทุกก้าวเล็กๆ ของน้องคือความสำเร็จของคุณแม่ ไม่ต้องกังวลไปนะคะ 🤍",
    "การเติบโตของทารกคลอดก่อนกำหนดต้องใช้เวลา คุณแม่ทำดีที่สุดแล้วค่ะ 🌼",
]
FIRST_ENTRY_MESSAGES = [
    "ยินดีต้อนรับสู่เส้นทางการเติบโตของน้องนะคะ คุณแม่เก่งมากที่ใส่ใจขนาดนี้ 💖",
]

# ============================================================================
# Sidebar: ฟอร์มกรอกข้อมูล — เรียบง่าย ที่สุด
# ============================================================================
with st.sidebar:
    st.markdown("### 📝 กรอกข้อมูลวันนี้")
    st.caption("กรอกทุกครั้งที่ชั่งน้ำหนัก/วัดตัวน้อง")

    with st.form("input_form", clear_on_submit=False):
        sex_label = st.radio("เพศของทารก", ["ชาย", "หญิง"], horizontal=True)
        ga_birth = st.number_input(
            "อายุครรภ์เมื่อแรกเกิด (สัปดาห์)",
            min_value=22.0, max_value=42.0, value=32.0, step=0.1,
        )
        corrected_age = st.number_input(
            "อายุครรภ์ปรับแก้ (สัปดาห์)",
            min_value=22.0, max_value=50.0, value=34.0, step=0.1,
        )
        current_weight = st.number_input(
            "น้ำหนักปัจจุบัน (กก.)",
            min_value=0.2, max_value=10.0, value=1.8, step=0.01, format="%.2f",
        )
        current_length = st.number_input(
            "ความยาวปัจจุบัน (ซม.)",
            min_value=20.0, max_value=70.0, value=42.0, step=0.1,
        )
        entry_date = st.date_input("วันที่บันทึก", value=datetime.today())

        submitted = st.form_submit_button("➕ บันทึกข้อมูล", use_container_width=True)

    if st.session_state.records:
        st.write("")
        if st.button("🗑️ ล้างข้อมูลทั้งหมด", use_container_width=True):
            st.session_state.records = []
            st.rerun()

sex_key = "boy" if sex_label == "ชาย" else "girl"

# ============================================================================
# บันทึกข้อมูล + ข้อความให้กำลังใจ
# ============================================================================
if submitted:
    new_record = {
        "date": entry_date,
        "sex": sex_key,
        "ga_birth": ga_birth,
        "corrected_age": corrected_age,
        "weight": current_weight,
        "length": current_length,
    }
    prev_records = [r for r in st.session_state.records if r["sex"] == sex_key]
    st.session_state.records.append(new_record)

    if not prev_records:
        st.success(np.random.choice(FIRST_ENTRY_MESSAGES))
    else:
        last = sorted(prev_records, key=lambda r: r["corrected_age"])[-1]
        weight_gain = current_weight - last["weight"]
        length_gain = current_length - last["length"]

        if weight_gain > 0:
            st.success(np.random.choice(WEIGHT_UP_MESSAGES))
            st.balloons()
        elif length_gain > 0:
            st.success(np.random.choice(LENGTH_UP_MESSAGES))
        else:
            st.info(np.random.choice(STEADY_MESSAGES))

# ============================================================================
# หน้าหลัก — เรียบง่าย กราฟน้ำหนักอยู่บนสุด ตามด้วยความยาว
# ============================================================================
st.markdown('<div class="big-title">🌷 การเติบโตของน้อง</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">ติดตามน้ำหนักและความยาว เทียบกับกราฟมาตรฐานสำหรับทารกคลอดก่อนกำหนด</div>',
    unsafe_allow_html=True,
)

records_df = pd.DataFrame(st.session_state.records)
sex_records_df = (
    records_df[records_df["sex"] == sex_key].sort_values("corrected_age")
    if not records_df.empty else pd.DataFrame()
)

# ---- กราฟน้ำหนัก: อยู่บนสุด เพราะสะท้อนผลของนมแม่ได้ไวที่สุด ----
fig_weight = plot_growth_chart(
    "weight", sex_key, "กก.", "น้ำหนัก", sex_records_df, "⚖️"
)
st.plotly_chart(fig_weight, use_container_width=True, config={"displayModeBar": False})

st.write("")

# ---- กราฟความยาว: อยู่ด้านล่าง ----
fig_length = plot_growth_chart(
    "length", sex_key, "ซม.", "ความยาว", sex_records_df, "📏"
)
st.plotly_chart(fig_length, use_container_width=True, config={"displayModeBar": False})

# ---- ประวัติข้อมูล (ย่อ เรียบง่าย) ----
with st.expander("📖 ดูประวัติการบันทึกข้อมูล"):
    if not sex_records_df.empty:
        display_df = sex_records_df.rename(columns={
            "date": "วันที่",
            "ga_birth": "อายุครรภ์แรกเกิด",
            "corrected_age": "อายุครรภ์ปรับแก้",
            "weight": "น้ำหนัก (กก.)",
            "length": "ความยาว (ซม.)",
        })[["วันที่", "อายุครรภ์แรกเกิด", "อายุครรภ์ปรับแก้", "น้ำหนัก (กก.)", "ความยาว (ซม.)"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.write("ยังไม่มีข้อมูล — กรอกข้อมูลในแถบด้านซ้ายเพื่อเริ่มติดตามค่ะ 💛")

st.divider()
st.caption(
    "⚠️ เส้นเปอร์เซ็นไทล์ในแอปนี้เป็นข้อมูลจำลองเพื่อสาธิตการทำงานเท่านั้น ไม่ใช่ตาราง "
    "Fenton ฉบับทางการ กรุณาใช้ร่วมกับคำแนะนำของกุมารแพทย์เสมอ · ไม่แสดงข้อมูลเส้นรอบศีรษะ "
    "เพื่อลดความกังวลใจของคุณแม่"
)

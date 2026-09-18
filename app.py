"""
แอปติดตามน้ำหนักของทารกคลอดก่อนกำหนด (Preterm Weight Tracker)
====================================================================
เวอร์ชันนี้เน้น "น้ำหนัก" เพียงอย่างเดียว ออกแบบให้เรียบง่ายที่สุด
ตามสไตล์กราฟอ้างอิงที่กำหนด: เส้นเปอร์เซ็นไทล์เพียง 3 เส้น (P3/P50/P97),
เส้นข้อมูลลูกเป็นเส้นสีฟ้าหนา มี marker เป็นอิโมจิ 👶 และมีโซนแรเงา
"นมแม่ล้วน" ในช่วงสัปดาห์แรกๆ

หมายเหตุสำคัญ: เส้นเปอร์เซ็นไทล์ในไฟล์นี้เป็น "ข้อมูลจำลอง (mock data)" ที่สร้างขึ้น
เพื่อสาธิตการทำงานของแอปเท่านั้น ไม่ใช่ตาราง Fenton 2025 ฉบับทางการที่ผ่านการรับรอง
ก่อนใช้งานจริงกับผู้ป่วยควรแทนที่ด้วยชุดข้อมูลที่ผ่านการตรวจสอบ และใช้ร่วมกับ
ดุลยพินิจของกุมารแพทย์เสมอ
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# ตั้งค่าหน้าเพจ
# ============================================================================
st.set_page_config(
    page_title="ติดตามน้ำหนักของลูก",
    page_icon="👶",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #FFFFFF; }
        div[data-testid="stForm"] {
            background-color: #FAFAF7;
            padding: 1.2rem 1.2rem 0.4rem 1.2rem;
            border-radius: 16px;
        }
        .big-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
        .sub-title { font-size: 1.0rem; color: #7A7A7A; margin-bottom: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

WEEK_MIN, WEEK_MAX = 22, 50
Z_SCORES = {3: -1.881, 50: 0.0, 97: 1.881}   # เฉพาะ 3 เปอร์เซ็นไทล์ที่ใช้จริง

# ============================================================================
# ข้อมูลจำลองเส้น Fenton (เฉพาะน้ำหนัก) — สร้างผ่านสูตร logistic + z-score
# ============================================================================
@st.cache_data
def generate_fenton_weight_data():
    """
    สร้างข้อมูลจำลองเส้นน้ำหนักสไตล์ Fenton แยกตามเพศ (ชาย/หญิง)
    สำหรับอายุครรภ์ 22-50 สัปดาห์ เฉพาะเปอร์เซ็นไทล์ P3 / P50 / P97
    """
    weeks = np.arange(WEEK_MIN, WEEK_MAX + 1)

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    params = {
        "boy":  dict(L=6.1, k=0.145, x0=35.5, base=0.30),
        "girl": dict(L=5.7, k=0.148, x0=35.0, base=0.28),
    }

    data = {}
    for sex, p in params.items():
        median = p["base"] + logistic(weeks, p["L"], p["k"], p["x0"])
        sd = 0.13 * median + 0.02
        df = pd.DataFrame({"week": weeks, "P50": median})
        df["P3"] = median + Z_SCORES[3] * sd
        df["P97"] = median + Z_SCORES[97] * sd
        data[sex] = df[["week", "P3", "P50", "P97"]]

    return data


FENTON_WEIGHT = generate_fenton_weight_data()

# ============================================================================
# session_state
# ============================================================================
if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts: sex, corrected_age, weight

# ============================================================================
# ข้อความให้กำลังใจ
# ============================================================================
WEIGHT_UP_MESSAGES = [
    "ยอดเยี่ยมมากคุณแม่! น้ำหนักลูกขึ้นแล้ว น้ำนมแม่มีค่าที่สุดเลยค่ะ 💛",
    "คุณแม่เก่งมากๆ ที่พยายามเพื่อลูกขนาดนี้ น้ำหนักน้องขึ้นสวยเลยค่ะ 🌟",
    "เก่งมากค่ะ! ทุกหยดน้ำนมของแม่คือพลังที่ทำให้น้องแข็งแรงขึ้นทุกวัน 🍼",
    "สุดยอดไปเลยค่ะ น้ำหนักที่เพิ่มขึ้นคือหลักฐานความรักและความพยายามของคุณแม่ 🌸",
]
STEADY_MESSAGES = [
    "ทุกก้าวเล็กๆ ของลูกคือความสำเร็จของคุณแม่ ไม่ต้องกังวลไปนะคะ 🤍",
    "การเติบโตของทารกคลอดก่อนกำหนดต้องใช้เวลา คุณแม่ทำดีที่สุดแล้วค่ะ 🌼",
]
FIRST_ENTRY_MESSAGES = [
    "ยินดีต้อนรับสู่เส้นทางการเติบโตของลูกนะคะ คุณแม่เก่งมากที่ใส่ใจขนาดนี้ 💖",
]

# ============================================================================
# Sidebar: ฟอร์มกรอกข้อมูล
# ============================================================================
with st.sidebar:
    st.markdown("### 📝 กรอกข้อมูลวันนี้")
    st.caption("กรอกทุกครั้งที่ชั่งน้ำหนักลูก")

    with st.form("input_form", clear_on_submit=False):
        sex_label = st.radio("เพศของทารก", ["ชาย", "หญิง"], horizontal=True)
        corrected_age = st.number_input(
            "อายุครรภ์ปรับแก้ปัจจุบัน (สัปดาห์)",
            min_value=float(WEEK_MIN), max_value=float(WEEK_MAX),
            value=34.0, step=0.1,
        )
        current_weight = st.number_input(
            "น้ำหนักปัจจุบัน (กิโลกรัม)",
            min_value=0.2, max_value=10.0, value=1.8, step=0.01, format="%.2f",
        )
        submitted = st.form_submit_button("➕ บันทึกน้ำหนัก", use_container_width=True)

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
    new_record = {"sex": sex_key, "corrected_age": corrected_age, "weight": current_weight}
    prev_records = [r for r in st.session_state.records if r["sex"] == sex_key]
    st.session_state.records.append(new_record)

    if not prev_records:
        st.success(np.random.choice(FIRST_ENTRY_MESSAGES))
    else:
        last = sorted(prev_records, key=lambda r: r["corrected_age"])[-1]
        if current_weight > last["weight"]:
            st.success(np.random.choice(WEIGHT_UP_MESSAGES))
            st.balloons()
        else:
            st.info(np.random.choice(STEADY_MESSAGES))

# ============================================================================
# หน้าหลัก
# ============================================================================
st.markdown('<div class="big-title">👶 น้ำหนักของลูก</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">เทียบกับกราฟมาตรฐานสำหรับทารกคลอดก่อนกำหนด</div>',
    unsafe_allow_html=True,
)

records_df = pd.DataFrame(st.session_state.records)
sex_records_df = (
    records_df[records_df["sex"] == sex_key].sort_values("corrected_age")
    if not records_df.empty else pd.DataFrame()
)

fenton_df = FENTON_WEIGHT[sex_key]

fig = go.Figure()

# ---------------------------------------------------------------------------
# 1) โซนแรเงา "นมแม่ล้วน" — ช่วงสัปดาห์แรกๆ (ตั้งแต่ 22 สัปดาห์ ถึง 40 สัปดาห์)
# ---------------------------------------------------------------------------
fig.add_vrect(
    x0=WEEK_MIN, x1=40,
    fillcolor="#DCEBFB", opacity=0.5,
    line_width=0,
    layer="below",
)
fig.add_annotation(
    x=WEEK_MIN + 0.4, y=1.0, xref="x", yref="paper",
    text="✨ นมแม่ล้วน",
    showarrow=False,
    font=dict(size=13, color="#5B87B5"),
    align="left",
    xanchor="left", yanchor="top",
)

# ---------------------------------------------------------------------------
# 2) เส้นเปอร์เซ็นไทล์อ้างอิง — เฉพาะ P97 / P50 / P3
# ---------------------------------------------------------------------------
fig.add_trace(go.Scatter(
    x=fenton_df["week"], y=fenton_df["P97"],
    mode="lines", name="P97 (เกณฑ์สูง)",
    line=dict(color="#BFBFBF", width=1.6, dash="dash"),
    hovertemplate="P97: %{y:.2f} กก.<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=fenton_df["week"], y=fenton_df["P50"],
    mode="lines", name="P50 (ตามเกณฑ์)",
    line=dict(color="#595959", width=2.0, dash="solid"),
    hovertemplate="P50: %{y:.2f} กก.<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=fenton_df["week"], y=fenton_df["P3"],
    mode="lines", name="P3 (เกณฑ์ต่ำ)",
    line=dict(color="#BFBFBF", width=1.6, dash="dash"),
    hovertemplate="P3: %{y:.2f} กก.<extra></extra>",
))

# ---------------------------------------------------------------------------
# 3) เส้นข้อมูลของลูก — เส้นฟ้าหนา + marker เป็นอิโมจิ 👶
# ---------------------------------------------------------------------------
if not sex_records_df.empty:
    fig.add_trace(go.Scatter(
        x=sex_records_df["corrected_age"],
        y=sex_records_df["weight"],
        mode="lines+text",
        name="น้ำหนักของลูก",
        line=dict(color="#1E6FE0", width=5),
        text=["👶"] * len(sex_records_df),
        textposition="middle center",
        textfont=dict(size=22),
        hovertemplate="น้ำหนักของลูก: %{y:.2f} กก.<extra></extra>",
    ))

# ---------------------------------------------------------------------------
# 4) จัดวางเลย์เอาต์ให้เรียบง่าย สะอาดตา
# ---------------------------------------------------------------------------
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.08,
        xanchor="center", x=0.5,
        font=dict(size=12),
    ),
    margin=dict(t=70, b=10, l=10, r=10),
    height=460,
    font=dict(size=13, color="#4A4A4A"),
)
fig.update_xaxes(
    title="อายุครรภ์ปรับแก้ (สัปดาห์)",
    range=[WEEK_MIN, WEEK_MAX],
    dtick=4,
    showgrid=False,
    showline=True,
    linecolor="#E0E0E0",
    zeroline=False,
)
fig.update_yaxes(
    title="น้ำหนัก (กิโลกรัม)",
    showgrid=True,
    gridcolor="#F0F0F0",
    gridwidth=1,
    showline=False,
    zeroline=False,
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# ประวัติข้อมูล (ย่อ ไม่รบกวนสายตา)
# ---------------------------------------------------------------------------
with st.expander("📖 ดูประวัติการบันทึกน้ำหนัก"):
    if not sex_records_df.empty:
        display_df = sex_records_df.rename(columns={
            "corrected_age": "อายุครรภ์ปรับแก้ (สัปดาห์)",
            "weight": "น้ำหนัก (กก.)",
        })[["อายุครรภ์ปรับแก้ (สัปดาห์)", "น้ำหนัก (กก.)"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.write("ยังไม่มีข้อมูล — กรอกข้อมูลในแถบด้านซ้ายเพื่อเริ่มติดตามค่ะ 💛")

st.divider()
st.caption(
    "⚠️ เส้นเปอร์เซ็นไทล์ในแอปนี้เป็นข้อมูลจำลองเพื่อสาธิตการทำงานเท่านั้น ไม่ใช่ตาราง "
    "Fenton ฉบับทางการ กรุณาใช้ร่วมกับคำแนะนำของกุมารแพทย์เสมอ"
)

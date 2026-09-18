"""
แอปติดตามการเติบโตของทารกคลอดก่อนกำหนด (Preterm Growth Tracker)
====================================================================
- ติดตามเฉพาะ "น้ำหนัก" และ "ความยาว" เท่านั้น (ไม่รวมเส้นรอบศีรษะโดยเจตนา
  เพื่อลดความกังวลของคุณแม่ และเน้นย้ำเรื่องการเจริญเติบโตพื้นฐาน + การกินนม)
- กราฟอ้างอิงจากแนวคิดของ Fenton Preterm Growth Chart (3rd generation)
- ข้อมูลเส้นเปอร์เซ็นไทล์ในไฟล์นี้เป็น "ข้อมูลจำลอง (mock data)" ที่สร้างขึ้น
  เพื่อสาธิตการทำงานของแอปเท่านั้น ไม่ใช่ตารางอ้างอิงทางคลินิกที่ผ่านการรับรอง
  ก่อนนำไปใช้จริงกับผู้ป่วย ควรแทนที่ด้วยชุดข้อมูล Fenton 2025 ฉบับทางการ
  และควรใช้ร่วมกับดุลยพินิจของกุมารแพทย์/บุคลากรทางการแพทย์เสมอ
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ----------------------------------------------------------------------------
# ตั้งค่าหน้าเพจ
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ติดตามการเติบโตทารกคลอดก่อนกำหนด",
    page_icon="🌱",
    layout="wide",
)

PERCENTILES = [3, 10, 50, 90, 97]
# ค่า z-score โดยประมาณของแต่ละเปอร์เซ็นไทล์ (การกระจายแบบปกติ) ใช้เพื่อสร้าง
# เส้นเปอร์เซ็นไทล์จำลองจากเส้นค่ากลาง (median) และค่าเบี่ยงเบนมาตรฐานโดยประมาณ
Z_SCORES = {3: -1.881, 10: -1.282, 50: 0.0, 90: 1.282, 97: 1.881}
WEEK_MIN, WEEK_MAX = 22, 50

PERCENTILE_COLORS = {
    3: "#f4a6a6",
    10: "#f7c59f",
    50: "#6fa8dc",
    90: "#f7c59f",
    97: "#f4a6a6",
}


# ----------------------------------------------------------------------------
# สร้างข้อมูลจำลองเส้นเปอร์เซ็นไทล์ (Fenton-style mock curves)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_fenton_mock_data():
    """
    สร้างข้อมูลจำลองเส้นโค้งการเติบโตสไตล์ Fenton แยกตามเพศ (ชาย/หญิง)
    สำหรับน้ำหนัก (กก.) และความยาว (ซม.) ตั้งแต่อายุครรภ์ 22-50 สัปดาห์

    หมายเหตุ: ใช้แบบจำลองเส้นโค้งลอจิสติก (logistic growth curve) เพื่อให้ได้รูปทรง
    คล้ายกราฟ Fenton จริง (เพิ่มขึ้นช้าช่วงต้น เร่งขึ้นช่วงกลาง คงตัวช่วงปลาย)
    แล้วใช้ z-score ของแต่ละเปอร์เซ็นไทล์คูณกับส่วนเบี่ยงเบนมาตรฐานโดยประมาณ
    บวกกับเส้นค่ากลาง เพื่อสร้างเส้นเปอร์เซ็นไทล์ 3/10/50/90/97
    """
    weeks = np.arange(WEEK_MIN, WEEK_MAX + 1)

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    data = {"boy": {}, "girl": {}}

    # ---------------- น้ำหนัก (กก.) ----------------
    # ค่ากลาง (P50) โดยประมาณ: ~0.45 กก. ที่ 22 สัปดาห์ -> ~3.4 กก. ที่ 40 สัปดาห์
    # -> ~5.4 กก. ที่ 50 สัปดาห์ (เด็กชายมักมีค่าเฉลี่ยสูงกว่าเด็กหญิงเล็กน้อย)
    weight_params = {
        "boy": dict(L=6.1, k=0.145, x0=35.5, base=0.30),
        "girl": dict(L=5.7, k=0.148, x0=35.0, base=0.28),
    }
    for sex, p in weight_params.items():
        median = p["base"] + logistic(weeks, p["L"], p["k"], p["x0"])
        # ส่วนเบี่ยงเบนมาตรฐานโดยประมาณ (เพิ่มขึ้นตามอายุครรภ์ ~13% ของค่ากลาง)
        sd = 0.13 * median + 0.02
        df = pd.DataFrame({"week": weeks, "P50": median})
        for pct in PERCENTILES:
            if pct == 50:
                continue
            df[f"P{pct}"] = median + Z_SCORES[pct] * sd
        df = df[["week"] + [f"P{p_}" for p_ in PERCENTILES]]
        data[sex]["weight"] = df

    # ---------------- ความยาว (ซม.) ----------------
    # ค่ากลาง (P50) โดยประมาณ: ~28 ซม. ที่ 22 สัปดาห์ -> ~50 ซม. ที่ 40 สัปดาห์
    # -> ~58 ซม. ที่ 50 สัปดาห์
    length_params = {
        "boy": dict(L=34.0, k=0.135, x0=34.5, base=24.5),
        "girl": dict(L=33.0, k=0.138, x0=34.0, base=24.0),
    }
    for sex, p in length_params.items():
        median = p["base"] + logistic(weeks, p["L"], p["k"], p["x0"])
        sd = 0.045 * median + 0.3
        df = pd.DataFrame({"week": weeks, "P50": median})
        for pct in PERCENTILES:
            if pct == 50:
                continue
            df[f"P{pct}"] = median + Z_SCORES[pct] * sd
        df = df[["week"] + [f"P{p_}" for p_ in PERCENTILES]]
        data[sex]["length"] = df

    return data


FENTON_DATA = generate_fenton_mock_data()


# ----------------------------------------------------------------------------
# session_state สำหรับเก็บข้อมูลที่คุณแม่กรอก
# ----------------------------------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = []  # list of dicts


# ----------------------------------------------------------------------------
# ฟังก์ชันวาดกราฟ
# ----------------------------------------------------------------------------
def plot_growth_chart(measure_key: str, sex: str, unit: str, title: str, records_df: pd.DataFrame):
    df = FENTON_DATA[sex][measure_key]
    fig = go.Figure()

    # เส้นเปอร์เซ็นไทล์พื้นหลัง
    for pct in PERCENTILES:
        col = f"P{pct}"
        fig.add_trace(
            go.Scatter(
                x=df["week"],
                y=df[col],
                mode="lines",
                name=f"เปอร์เซ็นไทล์ที่ {pct}",
                line=dict(
                    width=2.5 if pct == 50 else 1.3,
                    dash="solid" if pct == 50 else "dot",
                    color=PERCENTILE_COLORS[pct],
                ),
                hovertemplate=f"P{pct}: %{{y:.2f}} {unit}<br>อายุครรภ์: %{{x}} สัปดาห์<extra></extra>",
            )
        )

    # เส้นข้อมูลจริงของทารก (โดดเด่น สีเข้ม มี marker)
    if records_df is not None and not records_df.empty:
        fig.add_trace(
            go.Scatter(
                x=records_df["corrected_age"],
                y=records_df[measure_key],
                mode="lines+markers",
                name="น้องของคุณแม่ 💛",
                line=dict(color="#2e7d32", width=4),
                marker=dict(size=11, color="#2e7d32", symbol="star",
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"น้ำหนัก/ความยาวของน้อง: %{{y:.2f}} {unit}<br>"
                              f"อายุครรภ์ปรับแก้: %{{x}} สัปดาห์<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="อายุครรภ์ (สัปดาห์)",
        yaxis_title=f"{title.split('(')[0].strip()} ({unit})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=480,
        margin=dict(t=80),
    )
    fig.update_xaxes(range=[WEEK_MIN, WEEK_MAX], dtick=2)
    return fig


# ----------------------------------------------------------------------------
# ข้อความให้กำลังใจ
# ----------------------------------------------------------------------------
WEIGHT_UP_MESSAGES = [
    "กราฟสวยมาก! น้ำหนักหนูขึ้นแล้ว น้ำนมของคุณแม่วิเศษที่สุดเลยค่ะ 💛",
    "เก่งมากเลยคุณแม่! การเติบโตของหนูเป็นผลจากความรักและน้ำนมแม่นะคะ 🌱",
    "สุดยอดไปเลย! ทุกหยดน้ำนมของแม่คือพลังที่ทำให้หนูแข็งแรงขึ้นทุกวัน 🍼",
    "เยี่ยมมากค่ะ! น้ำหนักที่เพิ่มขึ้นคือหลักฐานว่าคุณแม่ทำได้ดีมากจริงๆ 🌟",
]
LENGTH_UP_MESSAGES = [
    "หนูตัวยาวขึ้นแล้วนะคะ! การเลี้ยงลูกด้วยนมแม่ช่วยให้หนูเติบโตได้เต็มศักยภาพ 🌷",
    "เก่งมาก! ความยาวตัวของหนูเพิ่มขึ้น แปลว่าโภชนาการจากนมแม่กำลังไปได้สวย 💪",
]
STEADY_MESSAGES = [
    "ทุกก้าวเล็กๆ ของหนูคือความสำเร็จของคุณแม่ อย่าเพิ่งกังวลไปนะคะ 🤍",
    "การเติบโตของทารกคลอดก่อนกำหนดต้องใช้เวลา คุณแม่ทำดีที่สุดแล้วค่ะ 🌼",
]
FIRST_ENTRY_MESSAGES = [
    "ยินดีต้อนรับสู่เส้นทางการเติบโตของน้องนะคะ คุณแม่เก่งมากที่ดูแลใส่ใจขนาดนี้ 💖",
]


# ----------------------------------------------------------------------------
# Sidebar: ฟอร์มกรอกข้อมูล
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 กรอกข้อมูลการเติบโตของน้อง")
    st.caption("กรอกข้อมูลทุกครั้งที่ชั่งน้ำหนัก/วัดความยาว เพื่อดูแนวโน้มการเติบโต")

    with st.form("input_form", clear_on_submit=False):
        sex_label = st.radio("เพศของทารก", ["ชาย", "หญิง"], horizontal=True)
        ga_birth = st.number_input(
            "อายุครรภ์เมื่อแรกเกิด (สัปดาห์)",
            min_value=22.0, max_value=42.0, value=32.0, step=0.1,
        )
        corrected_age = st.number_input(
            "อายุครรภ์ปรับแก้ปัจจุบัน (Corrected Age) (สัปดาห์)",
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
        entry_date = st.date_input("วันที่ชั่ง/วัด", value=datetime.today())

        submitted = st.form_submit_button("➕ บันทึกข้อมูลวันนี้", use_container_width=True)

    st.divider()
    if st.session_state.records:
        if st.button("🗑️ ล้างข้อมูลทั้งหมด", use_container_width=True):
            st.session_state.records = []
            st.rerun()

sex_key = "boy" if sex_label == "ชาย" else "girl"

# ----------------------------------------------------------------------------
# บันทึกข้อมูลใหม่ + ข้อความให้กำลังใจ
# ----------------------------------------------------------------------------
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

        if weight_gain > 0 and length_gain > 0:
            st.success(np.random.choice(WEIGHT_UP_MESSAGES))
            st.balloons()
        elif weight_gain > 0:
            st.success(np.random.choice(WEIGHT_UP_MESSAGES))
        elif length_gain > 0:
            st.success(np.random.choice(LENGTH_UP_MESSAGES))
        else:
            st.info(np.random.choice(STEADY_MESSAGES))

# ----------------------------------------------------------------------------
# หน้าหลัก
# ----------------------------------------------------------------------------
st.title("🌱 ติดตามการเติบโตของทารกคลอดก่อนกำหนด")
st.markdown(
    "แอปนี้ช่วยให้คุณแม่ติดตาม **น้ำหนัก** และ **ความยาว** ของน้อง เทียบกับกราฟ "
    "การเติบโตมาตรฐานสำหรับทารกคลอดก่อนกำหนด (แนวทาง Fenton) "
    "เพื่อเป็นกำลังใจในการให้นมแม่อย่างต่อเนื่องค่ะ 💛"
)

st.warning(
    "⚠️ **ข้อควรทราบ:** เส้นเปอร์เซ็นไทล์ในแอปนี้เป็น **ข้อมูลจำลอง (mock data)** "
    "ที่จัดทำขึ้นเพื่อสาธิตการทำงานของแอปเท่านั้น ยังไม่ใช่ตาราง Fenton ฉบับทางการ "
    "การประเมินการเติบโตของทารกที่แท้จริงควรทำร่วมกับกุมารแพทย์หรือบุคลากรทางการแพทย์เสมอ "
    "แอปนี้ไม่ได้ใช้แทนคำแนะนำทางการแพทย์"
)

records_df = pd.DataFrame(st.session_state.records)
sex_records_df = (
    records_df[records_df["sex"] == sex_key].sort_values("corrected_age")
    if not records_df.empty else pd.DataFrame()
)

col1, col2 = st.columns(2)
with col1:
    fig_weight = plot_growth_chart("weight", sex_key, "กก.", "น้ำหนัก (กก.) ตามอายุครรภ์", sex_records_df)
    st.plotly_chart(fig_weight, use_container_width=True)
with col2:
    fig_length = plot_growth_chart("length", sex_key, "ซม.", "ความยาว (ซม.) ตามอายุครรภ์", sex_records_df)
    st.plotly_chart(fig_length, use_container_width=True)

st.divider()
st.subheader("📖 ประวัติการบันทึกข้อมูล")
if not sex_records_df.empty:
    display_df = sex_records_df.rename(columns={
        "date": "วันที่",
        "ga_birth": "อายุครรภ์แรกเกิด (สัปดาห์)",
        "corrected_age": "อายุครรภ์ปรับแก้ (สัปดาห์)",
        "weight": "น้ำหนัก (กก.)",
        "length": "ความยาว (ซม.)",
    })[["วันที่", "อายุครรภ์แรกเกิด (สัปดาห์)", "อายุครรภ์ปรับแก้ (สัปดาห์)",
        "น้ำหนัก (กก.)", "ความยาว (ซม.)"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("ยังไม่มีข้อมูลบันทึกไว้ กรุณากรอกข้อมูลในแถบด้านซ้ายเพื่อเริ่มติดตามการเติบโตของน้องค่ะ 💛")

st.divider()
st.caption(
    "💡 หมายเหตุ: แอปนี้เน้นการเติบโตด้าน **น้ำหนัก** และ **ความยาว** เท่านั้น "
    "โดยไม่แสดงข้อมูลเส้นรอบศีรษะ เพื่อลดความกังวลใจของคุณแม่ และเน้นให้กำลังใจ "
    "ในการให้นมแม่อย่างต่อเนื่อง"
)

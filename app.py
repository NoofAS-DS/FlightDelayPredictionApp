import json
import numpy as np
import streamlit as st

@st.cache_data
def load_model_coeffs(path: str = "lr_coeffs.json"):
    with open(path, "r") as f:
        data = json.load(f)
    coeffs = np.array(data["coefficients"], dtype=float)
    intercept = float(data["intercept"])
    num_features = int(data.get("num_features", len(coeffs)))
    return coeffs, intercept, num_features


coeffs, intercept, num_features = load_model_coeffs()

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction")
st.caption("تطبيق لتقدير احتمال تأخر الرحلات الجوية بناءً على خصائص الرحلة.")

st.subheader("بيانات الرحلة")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input(
        "مسافة الرحلة (Miles)",
        min_value=0.0,
        max_value=6000.0,
        value=500.0,
        step=10.0
    )

    dep_delay = st.number_input(
        "تأخير الإقلاع (بالدقائق)",
        min_value=-60.0,
        max_value=600.0,
        value=0.0,
        step=1.0
    )

with col2:
    air_time = st.number_input(
        "مدة الطيران (Minutes)",
        min_value=10.0,
        max_value=1000.0,
        value=120.0,
        step=5.0
    )

feature_vector = np.array([distance, air_time, dep_delay], dtype=float)

if len(coeffs) < len(feature_vector):
    st.error("يوجد تعارض بين عدد الميزات وعدد معاملات النموذج. تحققي من ملف lr_coeffs.json.")
    st.stop()

used_coeffs = coeffs[: len(feature_vector)]


def predict_delay_proba(x_vec: np.ndarray) -> float:
    logit = float(np.dot(used_coeffs, x_vec) + intercept)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob


st.markdown("---")

if st.button("تقدير احتمال التأخر"):
    prob_delay = predict_delay_proba(feature_vector)
    prob_on_time = 1 - prob_delay

    st.subheader("نتيجة التقدير")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="احتمال تأخر الرحلة",
            value=f"{prob_delay:.2%}"
        )
    with col_b:
        st.metric(
            label="احتمال الوصول في الوقت",
            value=f"{prob_on_time:.2%}"
        )

    if prob_delay >= 0.5:
        st.error("الرحلة متوقع أن تتأخر.")
    else:
        st.success("الرحلة متوقع أن تصل في الوقت.")

st.markdown("---")

with st.expander("معلومات عن النموذج"):
    st.write(f"- عدد معاملات النموذج: **{num_features}**")
    st.write(f"- المعامل المستخدم في هذا النموذج المبسط: **{len(used_coeffs)}**")
    st.write(f"- قيمة الـ Intercept: `{intercept:.4f}`")


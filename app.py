import json
import numpy as np
import streamlit as st

# ==============================
# 1) تحميل معاملات النموذج
# ==============================
@st.cache_data
def load_model_coeffs(path: str = "lr_coeffs.json"):
    """
    تحميل معاملات نموذج Logistic Regression المصدّر من PySpark.
    الملف متوقع يحتوي:
    {
        "coefficients": [...],
        "intercept": ...,
        "num_features": ...
    }
    """
    with open(path, "r") as f:
        data = json.load(f)
    coeffs = np.array(data["coefficients"], dtype=float)
    intercept = float(data["intercept"])
    num_features = int(data.get("num_features", len(coeffs)))
    return coeffs, intercept, num_features


coeffs, intercept, num_features = load_model_coeffs()

# ==============================
# 2) إعداد صفحة Streamlit
# ==============================
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Delay Prediction App")
st.caption("نموذج مبني على معاملات Logistic Regression تم تدريبها في PySpark (Demo تعليمي).")

st.markdown(
    """
هذا التطبيق **تعليمي** يوضح فكرة:
1. تدريب نموذج على بيانات ضخمة باستخدام **PySpark**
2. استخراج **معاملات النموذج (coefficients + intercept)**
3. استخدام هذه المعاملات للتنبؤ داخل **Streamlit** بدون تشغيل Spark

> ⚠️ ملاحظة مهمة:  
> في التدريب الحقيقي، استخدمنا متجه ميزات كبير `features_scaled` يحتوي ميزات كثيرة  
> (distance, air_time, dep_delay, one-hot encoding, log features, ...).  
> هنا سنبني مثال مبسّط يستخدم 3 ميزات فقط للشرح.
"""
)

st.divider()

# ==============================
# 3) إدخال بيانات الرحلة من المستخدم
# ==============================
st.subheader("📥 أدخل بيانات الرحلة (مثال مبسّط)")

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
        "تأخير الإقلاع (بالدقائق، يمكن يكون سالب لو أقلعت بدري)",
        min_value=-60.0,
        max_value=600.0,
        value=0.0,
        step=1.0
    )

with col2:
    air_time = st.number_input(
        "مدة الطيران (Air Time - minutes)",
        min_value=10.0,
        max_value=1000.0,
        value=120.0,
        step=5.0
    )

st.info(
    "في نسخة PySpark الأصلية، هذه الميزات يتم تحويلها (Scaling + Encoding) "
    "ثم تدخل في متجه features_scaled. هنا نستخدمها كما هي كنسخة مبسّطة للتوضيح."
)

# ==============================
# 4) بناء متجه الميزات
# ==============================
# ⚠️ مهم: في التدريب الحقيقي، ترتيب الميزات داخل features_scaled يختلف.
# هنا نستخدم أول 3 معاملات من coeffs كمثال تعليمي:
# نفترض أنها تقابل [distance, air_time, dep_delay]
feature_vector = np.array([distance, air_time, dep_delay], dtype=float)

if len(coeffs) < len(feature_vector):
    st.error(
        f"عدد معاملات المودل ({len(coeffs)}) أقل من عدد الميزات في هذا المثال ({len(feature_vector)}).\n"
        "تأكدي من طريقة تصدير lr_coeffs.json أو قلّلي عدد الميزات."
    )
    st.stop()

used_coeffs = coeffs[: len(feature_vector)]

# ==============================
# 5) دالة التنبؤ
# ==============================
def predict_delay_proba(x_vec: np.ndarray) -> float:
    """
    حساب احتمال التأخير باستخدام:
    p = sigmoid(w · x + b)
    """
    logit = float(np.dot(used_coeffs, x_vec) + intercept)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob


# ==============================
# 6) زر التنبؤ
# ==============================
if st.button("🔮 توقّع احتمال تأخر الرحلة"):
    prob_delay = predict_delay_proba(feature_vector)
    prob_on_time = 1 - prob_delay

    st.subheader("🔎 نتيجة التنبؤ")

    st.metric(
        label="احتمال تأخر الرحلة (Delay Probability)",
        value=f"{prob_delay:.2%}"
    )

    st.metric(
        label="احتمال أن تكون في الوقت (On Time)",
        value=f"{prob_on_time:.2%}"
    )

    if prob_delay >= 0.5:
        st.error("❗ النموذج يتوقع أن **الرحلة متأخرة غالبًا**.")
    else:
        st.success("✔ النموذج يتوقع أن **الرحلة في الوقت غالبًا**.")

    st.caption(
        "هذا التنبؤ مبني على نسخة مبسّطة من الميزات، الهدف تعليمي وليس نظام حجز حقيقي."
    )

st.divider()

# ==============================
# 7) قسم اختياري: عرض معلومات عن النموذج
# ==============================
with st.expander("ℹ️ تفاصيل عن النموذج (للطلاب / المهتمين)"):
    st.write(f"🔢 عدد معاملات النموذج الكلي: **{num_features}**")
    st.write(f"📏 عدد المعاملات المستخدمة في هذا الـ Demo: **{len(used_coeffs)}**")
    st.write(f"⚙️ قيمة الـ Intercept: `{intercept:.4f}`")

    st.markdown(
        """
        **الفكرة العامة:**

        - تم تدريب النموذج باستخدام **PySpark LogisticRegression** على بيانات رحلات طيران.
        - بعد التدريب، تم استخراج:
            - المتجه `coefficients`
            - والقيمة `intercept`
        - تم حفظهم في ملف `lr_coeffs.json`.
        - التطبيق هنا يعيد استخدام نفس المعاملات للتنبؤ، بدون الحاجة إلى تشغيل Spark.
        """
    )

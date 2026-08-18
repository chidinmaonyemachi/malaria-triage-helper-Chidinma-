import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Malaria Symptom Triage Helper",
    page_icon="🦟",
    layout="centered"
)


# =====================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("malaria_triage_model.pkl")


model = load_model()


# =====================================================
# HEADER
# =====================================================

st.title("🦟 Malaria Symptom Triage Helper")

st.write(
    """
    This AI/ML educational prototype analyses basic patient
    information and self-reported symptoms to identify patterns
    associated with malaria in the training dataset.
    """
)

st.warning(
    "⚠️ Educational prototype only. This application does not "
    "diagnose malaria and should not replace professional medical "
    "assessment or malaria testing."
)


# =====================================================
# INPUT FORM
# =====================================================

with st.form("malaria_triage_form"):

    # -------------------------------------------------
    # PATIENT INFORMATION
    # -------------------------------------------------

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

    with col2:
        residence = st.selectbox(
            "Residence",
            ["Rural", "Urban"]
        )

        season = st.selectbox(
            "Season",
            ["Rainy", "Dry"]
        )

    uses_mosquito_net = st.checkbox(
        "Uses mosquito net"
    )


    # -------------------------------------------------
    # COMMON SYMPTOMS
    # -------------------------------------------------

    st.subheader("Symptoms")

    fever_days = st.number_input(
        "Number of days with fever",
        min_value=0,
        max_value=30,
        value=0
    )

    symptom_col1, symptom_col2 = st.columns(2)

    with symptom_col1:
        has_fever = st.checkbox("Fever")
        has_chills = st.checkbox("Chills / Rigors")
        has_headache = st.checkbox("Headache")

    with symptom_col2:
        has_vomiting = st.checkbox("Vomiting")
        has_diarrhea = st.checkbox("Diarrhea")
        has_weakness = st.checkbox("Weakness / Fatigue")


    # -------------------------------------------------
    # DANGER SIGNS
    # -------------------------------------------------

    st.subheader("Danger Signs")

    st.caption(
        "Select any serious warning signs currently being experienced."
    )

    danger_col1, danger_col2 = st.columns(2)

    with danger_col1:
        difficulty_breathing = st.checkbox(
            "Difficulty breathing"
        )

        confusion = st.checkbox(
            "Confusion / impaired consciousness"
        )

        seizures = st.checkbox(
            "Multiple seizures / convulsions"
        )

    with danger_col2:
        dark_bloody_urine = st.checkbox(
            "Dark or bloody urine"
        )

        jaundice = st.checkbox(
            "Yellowing of eyes or skin"
        )

        abnormal_bleeding = st.checkbox(
            "Abnormal bleeding"
        )


    # -------------------------------------------------
    # SUBMIT BUTTON
    # -------------------------------------------------

    submitted = st.form_submit_button(
        "Check Symptoms",
        type="primary",
        use_container_width=True
    )


# =====================================================
# PROCESS RESULT
# =====================================================

if submitted:

    st.divider()

    st.subheader("Triage Result")


    # -------------------------------------------------
    # BASIC INPUT VALIDATION
    # -------------------------------------------------

    inconsistent_fever = (
        (has_fever and fever_days == 0)
        or
        (not has_fever and fever_days > 0)
    )

    if inconsistent_fever:

        st.warning(
            "Please check the fever information. "
            "The fever checkbox and number of fever days "
            "do not appear to match."
        )

        st.stop()


    # -------------------------------------------------
    # CHECK FOR DANGER SIGNS FIRST
    # -------------------------------------------------

    danger_signs = [
        difficulty_breathing,
        confusion,
        seizures,
        dark_bloody_urine,
        jaundice,
        abnormal_bleeding
    ]

    danger_detected = any(danger_signs)


    if danger_detected:

        st.error(
            "URGENT MEDICAL ASSESSMENT RECOMMENDED"
        )

        st.write(
            """
            One or more serious warning signs were reported.
            These symptoms require urgent medical assessment.
            """
        )

        st.warning(
            "Please seek emergency or urgent medical care. "
            "Do not rely on the machine-learning prediction "
            "when serious warning signs are present."
        )

        st.info(
            "The ML model has not been used for this result. "
            "The safety rule takes priority over the prediction model."
        )


    # -------------------------------------------------
    # USE MACHINE LEARNING MODEL
    # -------------------------------------------------

    else:

        patient_data = pd.DataFrame({
            "age_years": [age],
            "sex": [sex],
            "residence": [residence],
            "season": [season],
            "uses_mosquito_net": [uses_mosquito_net],
            "fever_days": [fever_days],
            "has_fever": [has_fever],
            "has_chills": [has_chills],
            "has_headache": [has_headache],
            "has_vomiting": [has_vomiting],
            "has_diarrhea": [has_diarrhea],
            "has_weakness": [has_weakness]
        })


        # Make prediction
        prediction = model.predict(patient_data)[0]


        # Get probability associated with positive class
        probability = model.predict_proba(
            patient_data
        )[0][1]


        # ---------------------------------------------
        # POSITIVE PATTERN
        # ---------------------------------------------

        if prediction == 1:

            st.error(
                "Malaria-like symptom pattern detected."
            )

            st.write(
                """
                The machine-learning model identified a symptom
                pattern similar to malaria-positive examples in
                its synthetic training dataset.
                """
            )

            st.info(
                "Recommended action: Seek appropriate medical "
                "assessment and malaria testing."
            )


        # ---------------------------------------------
        # NEGATIVE PATTERN
        # ---------------------------------------------

        else:

            st.success(
                "Malaria-like symptom pattern was not strongly detected."
            )

            st.write(
                """
                The model did not identify a strong malaria-associated
                pattern from the information entered.
                """
            )

            st.info(
                "This result does not rule out malaria or another "
                "illness. Seek medical assessment if symptoms persist "
                "or worsen."
            )


        # ---------------------------------------------
        # MODEL SCORE
        # ---------------------------------------------

        st.metric(
            "ML Pattern Score",
            f"{probability * 100:.1f}%"
        )

        st.caption(
            "This score represents the output of the educational "
            "machine-learning model. It is not a clinical probability "
            "and must not be interpreted as the chance that a person "
            "has malaria."
        )


# =====================================================
# ABOUT THE PROJECT
# =====================================================

st.divider()

with st.expander("ℹ️ About this project"):

    st.markdown(
        """
        ### Project Purpose

        The Malaria Symptom Triage Helper is a student AI/ML
        project demonstrating how machine learning can be used
        to analyse symptom patterns.

        ### Machine Learning Approach

        Two classification algorithms were evaluated:

        - Logistic Regression
        - Random Forest

        Random Forest was selected as the final model after
        comparison using test-set evaluation and five-fold
        cross-validation.

        ### Model Inputs

        The model uses:

        - Age
        - Sex
        - Residence
        - Season
        - Mosquito-net usage
        - Fever duration
        - Fever
        - Chills
        - Headache
        - Vomiting
        - Diarrhea
        - Weakness / fatigue

        ### Important Limitation

        The model was trained using synthetic malaria data.

        The synthetic dataset contains unusually strong differences
        between malaria-positive and malaria-negative cases.
        Therefore, the very high model performance observed during
        evaluation must not be interpreted as real-world clinical
        performance.

        ### Responsible AI

        This system is intended for educational demonstration only.

        It does not diagnose malaria.

        Laboratory testing and assessment by qualified healthcare
        professionals remain necessary for diagnosis.
        """
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🦟 Malaria Symptom Triage Helper | AI/ML Student Project"
)

import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Admission Chance Predictor", layout="centered")
st.title("Admission Chance Neural Network App")
st.markdown("This app predicts whether an applicant has a high admission chance using a neural network model.")
st.write("Model used: MLPClassifier")
st.write("Current test accuracy: 85.00%")

with open("models/admission_mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


def build_input_df():
    gre = st.slider("GRE Score", 260, 340, 320)
    toefl = st.slider("TOEFL Score", 0, 120, 105)
    university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=3)
    sop = st.slider("SOP Strength", 1.0, 5.0, 4.0, 0.5)
    lor = st.slider("LOR Strength", 1.0, 5.0, 4.0, 0.5)
    cgpa = st.slider("CGPA", 0.0, 10.0, 8.5, 0.1)
    research = st.selectbox("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    input_dict = {
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "University_Rating_1": 1 if university_rating == 1 else 0,
        "University_Rating_2": 1 if university_rating == 2 else 0,
        "University_Rating_3": 1 if university_rating == 3 else 0,
        "University_Rating_4": 1 if university_rating == 4 else 0,
        "University_Rating_5": 1 if university_rating == 5 else 0,
        "Research_0": 1 if research == 0 else 0,
        "Research_1": 1 if research == 1 else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df, gre, toefl, university_rating, sop, lor, cgpa, research


def get_rule_based_explanation(gre, toefl, university_rating, sop, lor, cgpa, research):
    strengths = []
    concerns = []

    if gre >= 320:
        strengths.append("strong GRE score")
    elif gre < 300:
        concerns.append("lower GRE score")

    if toefl >= 105:
        strengths.append("strong TOEFL score")
    elif toefl < 95:
        concerns.append("lower TOEFL score")

    if cgpa >= 8.5:
        strengths.append("high CGPA")
    elif cgpa < 7.5:
        concerns.append("lower CGPA")

    if research == 1:
        strengths.append("research experience")
    else:
        concerns.append("no research experience")

    if university_rating >= 4:
        strengths.append("good university rating")
    elif university_rating <= 2:
        concerns.append("lower university rating")

    if sop >= 4.0:
        strengths.append("strong SOP")
    elif sop <= 2.5:
        concerns.append("weaker SOP")

    if lor >= 4.0:
        strengths.append("strong LOR")
    elif lor <= 2.5:
        concerns.append("weaker LOR")

    return strengths, concerns


def get_improvement_suggestions(gre, toefl, cgpa, research, sop, lor):
    suggestions = []

    if gre < 320:
        suggestions.append("Improve GRE score to strengthen academic competitiveness.")
    if toefl < 105:
        suggestions.append("Improve TOEFL score to demonstrate stronger English proficiency.")
    if cgpa < 8.5:
        suggestions.append("Increase CGPA if possible or highlight strong academic consistency.")
    if research == 0:
        suggestions.append("Gain research experience through projects, assistantships, or publications.")
    if sop < 4.0:
        suggestions.append("Strengthen your Statement of Purpose to better communicate goals and fit.")
    if lor < 4.0:
        suggestions.append("Obtain stronger Letters of Recommendation from academic or professional mentors.")

    return suggestions


def get_confidence_level(probability):
    if probability >= 0.85:
        return "Very Strong Profile"
    elif probability >= 0.65:
        return "Moderate Chance"
    else:
        return "Needs Improvement"


def get_university_tier(probability):
    if probability >= 0.85:
        return "Top-tier universities may be realistic options."
    elif probability >= 0.65:
        return "Mid-tier universities may be a balanced target."
    else:
        return "Safer university options may be more suitable at this stage."


def simulate_profile_change(input_df, field_name, new_value):
    temp_df = input_df.copy()

    if field_name in temp_df.columns:
        temp_df[field_name] = new_value

    temp_scaled = scaler.transform(temp_df)
    temp_prob = model.predict_proba(temp_scaled)[0][1]
    return temp_prob


input_df, gre, toefl, university_rating, sop, lor, cgpa, research = build_input_df()

if st.button("Predict Admission Outcome"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    strengths, concerns = get_rule_based_explanation(
        gre, toefl, university_rating, sop, lor, cgpa, research
    )
    suggestions = get_improvement_suggestions(
        gre, toefl, cgpa, research, sop, lor
    )
    confidence_level = get_confidence_level(probability)
    tier_suggestion = get_university_tier(probability)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("High Admission Chance")
    else:
        st.error("Lower Admission Chance")

    st.write(f"Probability of High Admission Chance: {probability:.2%}")
    st.info(confidence_level)

    st.markdown("---")
    st.subheader("Interpretation")

    if strengths:
        st.write("Strengths in this profile:")
        for item in strengths:
            st.write(f"- {item}")

    if concerns:
        st.write("Areas of concern:")
        for item in concerns:
            st.write(f"- {item}")

    st.markdown("---")
    st.subheader("Suggested University Tier")
    st.write(tier_suggestion)

    st.markdown("---")
    st.subheader("How to Improve Your Profile")
    if suggestions:
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("Your profile already appears strong across the main evaluated factors.")

    st.markdown("---")
    st.subheader("What-If Scenario Analysis")

    gre_boost = simulate_profile_change(input_df, "GRE_Score", max(gre, 330))
    toefl_boost = simulate_profile_change(input_df, "TOEFL_Score", max(toefl, 110))
    cgpa_boost = simulate_profile_change(input_df, "CGPA", max(cgpa, 9.0))

    research_boost_df = input_df.copy()
    if "Research_0" in research_boost_df.columns:
        research_boost_df["Research_0"] = 0
    if "Research_1" in research_boost_df.columns:
        research_boost_df["Research_1"] = 1
    research_boost = model.predict_proba(scaler.transform(research_boost_df))[0][1]

    st.write(f"- If GRE improves to {max(gre, 330)}, probability becomes: **{gre_boost:.2%}**")
    st.write(f"- If TOEFL improves to {max(toefl, 110)}, probability becomes: **{toefl_boost:.2%}**")
    st.write(f"- If CGPA improves to {max(cgpa, 9.0):.1f}, probability becomes: **{cgpa_boost:.2%}**")
    st.write(f"- If research experience is added, probability becomes: **{research_boost:.2%}**")

    st.markdown("---")
    st.subheader("Input Summary")
    st.write(f"**GRE Score:** {gre}")
    st.write(f"**TOEFL Score:** {toefl}")
    st.write(f"**University Rating:** {university_rating}")
    st.write(f"**SOP:** {sop}")
    st.write(f"**LOR:** {lor}")
    st.write(f"**CGPA:** {cgpa}")
    st.write(f"**Research Experience:** {'Yes' if research == 1 else 'No'}")
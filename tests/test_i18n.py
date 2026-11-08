from wayfare.explain import explain_pair, render_explanation
from wayfare.i18n import catalogue_keys, t


def test_chinese_and_english_catalogues_match():
    english, chinese = catalogue_keys()
    assert english == chinese
    assert t("nav.sample", "zh") != t("nav.sample", "en")
    assert "示例" in t("nav.sample", "zh")
    assert "sample" in t("nav.sample", "en").lower()


def test_explanation_switches_language_without_changing_the_factor():
    alphas = {"preference_match": 1, "travel_time": 0, "cost": 0, "crowd": 0}
    features = {"preference_match": 0.9, "travel_time": 0.2, "cost": 0.2, "crowd": 0.2}
    explanation = explain_pair(
        "park",
        "Park",
        [features],
        [1],
        alphas,
        alternative_id="mall",
        alternative_name="Mall",
        alternative_rows=[{"preference_match": 0.1, "travel_time": 0.2, "cost": 0.2, "crowd": 0.2}],
    )
    assert explanation.top_factor == "preference_match"
    chinese_why, chinese_not, _ = render_explanation(explanation, "zh")
    english_why, english_not, _ = render_explanation(explanation, "en")
    assert "偏好匹配" in chinese_why
    assert "preference match" in english_why
    assert "偏好匹配" in chinese_not
    assert "preference match" in english_not
    assert explanation.top_factor == "preference_match"


def test_ui_language_switch_changes_the_sample_button():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=40).run()
    labels = " ".join(button.label for button in app.button)
    assert "载入新加坡示例行程" in labels
    app.radio(key="lang").set_value("en").run()
    labels = " ".join(button.label for button in app.button)
    assert "Load sample Singapore trip" in labels
    assert app.session_state["lang"] == "en"

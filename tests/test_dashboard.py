from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_investigator_dashboard_renders() -> None:
    app = AppTest.from_file(
        APP_PATH,
        default_timeout=30,
    ).run()

    assert not app.exception
    assert len(app.metric) >= 5
    assert len(app.dataframe) >= 1
    assert len(app.tabs) == 3
    assert len(app.file_uploader) == 1
    assert len(app.toggle) == 1
"""Launch the actual Streamlit entrypoint in Streamlit's headless test harness."""
from streamlit.testing.v1 import AppTest
from pathlib import Path
APP = Path(__file__).resolve().parents[1] / "app.py"

def test_command_center_streamlit_renders():
    app=AppTest.from_file(APP,default_timeout=60).run()
    assert len(app.exception)==0, str([str(x.message) for x in app.exception])
    assert len(app.metric)>=4
    assert any("Traffic" in str(x) or "Command" in str(x) for x in app.markdown)

def test_model_arena_streamlit_renders():
    app=AppTest.from_file(APP,default_timeout=90).run()
    radio=app.radio[0]
    radio.set_value("▥  Model arena").run()
    assert len(app.exception)==0, str([str(x.message) for x in app.exception])
    assert len(app.dataframe)>=1

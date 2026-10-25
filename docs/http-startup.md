# Streamlit server health verification

The routine GitHub Actions suite checks compilation, 10 Python/Streamlit tests, and actual Streamlit HTTP server startup on port 8501 (`/_stcore/health`). This tests a temporary GitHub Actions runner, not a permanent Codespace. The real iPinYou archive loader was also tested separately against the authentic ZIP and its download workflow is now opt-in.

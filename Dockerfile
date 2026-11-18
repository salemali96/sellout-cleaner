# Hugging Face Docker Space (also runs anywhere Docker runs). Hosted mode: each visitor gets a private folder
# and private AI settings; the host's key, if any, comes from the Space's secrets, never from a file.
FROM python:3.12-slim

RUN pip install --no-cache-dir "uv>=0.9,<0.14" && useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    UV_PROJECT_ENVIRONMENT=/home/user/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/home/user/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    SELLOUT_HOSTED=1
WORKDIR /home/user/app

# dependencies first, so code changes rebuild in seconds
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .

EXPOSE 7860
# Spaces show the app inside an iframe, where the browser blocks the cookie Streamlit's XSRF check needs, so
# uploads would fail with 403. Turned off here; each visitor's files are still private to their own session.
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]

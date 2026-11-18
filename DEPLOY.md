# Deploy the Sellout Cleaner

## Free: Streamlit Community Cloud

You get a link like `https://<name>.streamlit.app`. Free, about 1 GB memory per app: fine for files up to
roughly 100k rows (bigger files may hit the memory limit). The app sleeps when unused and wakes on the next visit.

1. Push this folder to a public GitHub repository (`.gitignore` keeps out `.env`, `.venv` and your data).
2. Sign in at https://share.streamlit.io with GitHub, then **Create app → Deploy a public app from GitHub**:
   - Repository: `<github-user>/sellout-cleaner`, branch `main`, main file `app.py`
   - **Advanced settings → Python version: 3.12**
3. Click **Deploy**. The first build takes a few minutes.

Hosted mode switches on by itself there: each visitor gets a private folder and private AI settings, and pastes
their own (free) key in the sidebar. Rules-only cleaning needs no key.

Optional shared key: in the app's **Settings → Secrets**, add for example
```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your key"
```
Everyone then uses your free daily limit.

To update: commit and `git push`; the app redeploys automatically.

## Paid: Hugging Face Spaces (PRO)

> **Not free (checked Nov 2026):** Hugging Face now requires a PRO subscription to host Docker (and Gradio)
> Spaces, even on cpu-basic hardware. Only static Spaces are free. Creating the Space fails with
> "402 Payment Required" without PRO.

You get a public link like `https://huggingface.co/spaces/<your-name>/sellout-cleaner` that anyone can open.
The free hardware (2 CPUs, 16 GB memory) handles files of 500k+ rows.

### 1. Create the Space (once)

1. Sign in at https://huggingface.co (free account).
2. Go to https://huggingface.co/new-space
   - **Space name:** `sellout-cleaner`
   - **SDK:** Docker → Blank
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public (anyone with the link) or Private (only you and people in your organization)
3. Create a token for uploading: https://huggingface.co/settings/tokens → *Create new token* → type **Write**.

### 2. Add your shared AI key (optional)

On the Space page: **Settings → Variables and secrets**.

| Type | Name | Value |
|---|---|---|
| Variable | `LLM_PROVIDER` | `gemini` (or `groq`, `huggingface`, ...) |
| Secret | `GEMINI_API_KEY` | your key (or `GROQ_API_KEY`, `HF_TOKEN`, ...) |

- With a shared key, visitors can run the AI agents right away. Everyone uses **your** free daily limit,
  so anyone with the link can use it up. Make the Space private if that matters.
- Without one, each visitor pastes their own free key in the sidebar.
- Either way, a visitor's own key is kept only in their browser session: never saved on the server, never
  visible to other visitors. Secrets are never shown to visitors or written into the code.

### 3. Upload the code

From the project folder, in a terminal (replace `<your-name>`):

```bash
git init -b main                      # skip if the folder is already a git repo
git add .                             # .gitignore keeps out .env, .venv and your data
git status                            # check: no .env and no data files in the list
git commit -m "Sellout Cleaner"
git remote add space https://huggingface.co/spaces/<your-name>/sellout-cleaner
git push --force space main           # user name = your HF name, password = the Write token
```

The Space builds for a few minutes (**Logs** tab), then the app opens on the Space page.

To update later: `git add . && git commit -m "update" && git push space main`.

### Good to know

- A free Space sleeps after about 48 hours without visitors; the next visitor waits ~1 minute for it to wake.
- Files are kept only while the Space runs, so visitors should download their results.
- *Ollama* and *Custom URL* providers are hidden on the hosted app (there is no local model on the server, and
  the server should not call addresses typed in by visitors).
- Your local `.env` is never uploaded: it is in `.gitignore` and `.dockerignore`.

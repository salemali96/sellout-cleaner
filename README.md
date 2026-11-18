---
title: Sellout Cleaner
emoji: 🧹
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Messy sellout CSV in, clean tables and a quarantine out
---

# Sellout Cleaner

Turns a large, messy retailer sell-out CSV into four clean tables (product, market, promo, weekly sales), a
quarantine of bad rows with the reason for each, a data quality report and a proof that no row, unit or value
was lost. Plain code does the checking and fixing; AI agents make the judgment calls; risky fixes wait for you.

- **Clean (rules only)** needs no AI key.
- **AI agents** and **Ask your data** need an AI provider. Several are free (Gemini, Groq, Hugging Face,
  OpenRouter, Mistral). Your key stays in your browser session only.
- Uploaded files and results are private to your session and deleted after 24 hours.

Run it locally: `uv sync && uv run streamlit run app.py`. Deploying: see [DEPLOY.md](DEPLOY.md).

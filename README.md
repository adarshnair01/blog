# Adarsh Nair - Data Science Portfolio

<div align="center">

**Senior Data Science Manager | ML Engineer**

[View Portfolio](https://adarshnair.online/blog)

---

[![Deploy site](https://github.com/adarshnair01/blog/actions/workflows/deploy.yml/badge.svg)](https://github.com/adarshnair01/blog/actions/workflows/deploy.yml)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-adarshnair01-blue?logo=linkedin)](https://www.linkedin.com/in/adarshnair01/)
[![GitHub](https://img.shields.io/badge/GitHub-adarshnair01-black?logo=github)](https://github.com/adarshnair01)

</div>

## 🚀 About This Repo

This repository contains my personal portfolio and blog, built using the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme. It showcases my professional experience, data science projects, and technical writings.

### Key Sections:
- **[Blog](https://adarshnair.online/blog/)**: Deep dives into Machine Learning, AI, and Data Science.
- **[Projects](https://adarshnair.online/blog/projects/)**: Detailed write-ups of internal and open-source data science projects.
- **[CV](https://adarshnair.online/blog/cv/)**: My professional resume, including experience at Yum! Digital & Technology and Kvantum Inc.
- **[Repositories](https://adarshnair.online/blog/repositories/)**: Key open-source projects from my GitHub.

---

## 🛠 Featured Projects

| Project | Description | Year |
| :--- | :--- | :--- |
| **Fine-tuning LLM for Intents** | Custom LLaMA 3/Mistral fine-tuning for high-accuracy intent recognition. | 2026 |
| **WhatsApp Chatbot using LLMs** | Enterprise-grade RAG-based AI support bot integrated with WhatsApp. | 2025 |
| **Recommendation Engine** | Two-Tower TFRS architecture for scalable product personalization. | 2022 |
| **Forecasting for Pharma Client** | Hierarchical time series SKU forecasting reducing inventory waste. | 2020 |
| **Voice of Consumer** | NLP-powered sentiment and trend analysis for social listening. | 2019 |

---

## 🚀 Deployment & Development

This site is deployed to **GitHub Pages** using **GitHub Actions**.

### Local Development

To run the site locally, you need Ruby and Bundler installed:

```bash
# Install dependencies
bundle install

# Run the local development server
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000/blog/`.

### Deployment to GitHub Pages

Deployment is automated via GitHub Actions on every push to the `master` branch.

1.  **Push Changes**:
    ```bash
    git add .
    git commit -m "Your update message"
    git push origin master
    ```
2.  **Workflow**: The `deploy.yml` action will:
    - Build the Jekyll site (including PDF CV generation).
    - Optimize assets.
    - Push the generated static files to the `gh-pages` branch.
3.  **URL**: The site is live at [https://adarshnair.online/blog/](https://adarshnair.online/blog/).

---

## 🎨 Acknowledgments

This site is based on the excellent [al-folio](https://github.com/alshedivat/al-folio) theme. 

## ⚖️ License

All content is © 2026 Adarsh Nair. The code is available under the MIT License.

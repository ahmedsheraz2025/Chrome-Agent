# Multi-Agent Browser Task System

A Python-based multi-agent system that translates user queries into clear English and performs browser tasks like searching or visting any site using Gemini AI model.

## 📖 Overview

This project is a modular, asynchronous agent system designed to process user input and execute browser-based tasks. It consists of two primary agents:

1. **Translator Agent**: Converts user input (in any language, e.g., Hindi, English) into clear, grammatically correct English.
2. **Browser Agent**: Executes browser tasks, such as performing web searches based on the translated query.

The system leverages the power of Google's Gemini AI (`gemini-1.5-flash`) for natural language processing and integrates with a custom `BrowserAgent` for browser interactions. It uses Chainlit as the front-end interface, providing an interactive, real-time user experience for inputting queries and viewing task results. The project is built with scalability in mind, allowing developers to extend it with additional agents or tools.

## 🚀 Features

- **Language Translation**: Converts user queries from any language to clear, concise English.
- **Browser Automation**: Performs web searches based on user input.
- **Asynchronous Processing**: Handles tasks efficiently using Python's `asyncio` for non-blocking operations.
- **Modular Design**: Easily extendable with new agents or tools for additional functionality.
- **Environment Configuration**: Securely manages API keys using a `.env` file.
- **Interactive Interface**: - Powered by Chainlit, enabling real-time query input and streamed task results.

## 📬 Contact

For issues or suggestions, open a GitHub issue or email [ahmedsheraz2025@gmail.com](mailto:ahmedsheraz2025@gmail.com).

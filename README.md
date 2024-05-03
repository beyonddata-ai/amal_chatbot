# 2024 AI Challenge - Amal
# Amal
## Table of contents
* [Introduction] (#introduction)
* [Demo] (#demo)
* [Architecture] (#architecture)
* [Features] (#features)
* [Technologies] (#technologies)
* [Setup] (#setup)

## Introduction
AMAL - “Automated Mentor for Aspiring to Leaders” is envisioned as a GPT style business
coach whose major function is to provide business advice and emotional support to women
entrepreneurs throughout their entrepreneurial journey.

## Demo
![alt text](https://github.com/beyonddata-ai/amal_chatbot/blob/main/amal.mp4?raw=true)

# High Level Architecture
![alt text](https://github.com/beyonddata-ai/amal_chatbot/blob/main/amal.png?raw=true)

# Local Setup
* 1 - Clone the repository.
* 2 - Install the requirements.txt file with command 'pip install -r requirements.txt' (Recommended: create separate environment to avoid any conflict)
* 3 - Give your OPENAI api key and google Gemini api key to get access the GPT model inference. (in './auth/keys.json')
* 4 - Give your google translation api key and other parameters to get access to translation model inference (in './auth/ornate-bebop-415006-4025a20e8711)
* 5 - Run the project with command 'python app.py'
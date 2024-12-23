# Shiki Project

## Overview
Shiki is an AI-powered automation framework utilizing Qwen2-VL for coordination inference and LangGraph for workflow automation. It enables intelligent interaction with business applications.

## Structure
- `server/`: FastAPI server for handling Qwen2-VL model inference.
- `client/`: LangGraph-based automation client for executing workflows.
- `env/`: Environment setup files.

## Quick Start
1. Set up the environment:
   ```bash
   conda env create -f env/environment.yml
   conda activate shiki


# teameeu-ai
- fastapi AI webapp
# Environment setting
- Create `.env` file
  - env example
  ```
  OPENAI_API_KEY=your-key
  OPENAI_API_MODEL=gpt-5-chat-latest
  CAREERNET_API_KEY=your-key
  HF_TOKEN=your-key
  ```
- Install libraries
  ```
  pip install -r requirements.txt
  ```
- Create jsonl files
  ```
  python data/careernet_depts.py
  python data/careernet_jobs.py
  ```
# Run App
- Run `app/main.py`
```
python -m main.app
```

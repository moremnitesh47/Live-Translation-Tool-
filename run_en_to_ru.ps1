# PowerShell launcher for EN->RU
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\live_translate.py --src en --tgt ru --whisper small --block-ms 20 --silence-ms 350 --vad-aggr 2 --glossary glossary.ru.json $args

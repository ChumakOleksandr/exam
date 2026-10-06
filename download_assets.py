"""Завантажити початкові слайди (потрібен інтернет)."""
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent

def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=90) as source:
        temporary.write_bytes(source.read())
    temporary.replace(target)
    print(target.name)

def main():
    url = 'https://api.github.com/repos/HalyshAnton/ITStep-AI/contents/data/exam/hands/slides?ref=exam'
    with urllib.request.urlopen(url, timeout=30) as response:
        files = json.load(response)
    for item in files:
        if item['type'] == 'file' and item['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
            download(item['download_url'], ROOT / 'slides' / Path(item['name']).name)

if __name__ == '__main__':
    main()

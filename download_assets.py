"""Повторне завантаження відсутніх слайдів із джерела в YAML."""

import argparse
import json
from pathlib import Path
import urllib.request

from config.config import DEFAULT_CONFIG, load_config


def download(url, target, timeout):
    """Завантажити файл атомарно, не перезаписуючи наявний.

    Args:
        url (str): Адреса зображення для завантаження.
        target (pathlib.Path): Шлях, за яким потрібно зберегти файл.
        timeout (float): Максимальний час очікування мережевої операції в секундах.

    Returns:
        None: Функція не повертає значення.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=timeout) as source:
        temporary.write_bytes(source.read())

    temporary.replace(target)
    print(target.name)


def parse_args(argv=None):
    """Прочитати шлях повної конфігурації для завантаження.

    Args:
        argv (list[str] | None): Аргументи командного рядка; None читає sys.argv.

    Returns:
        argparse.Namespace: Розібрані параметри командного рядка.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Path to full YAML configuration')

    return parser.parse_args(argv)


def main(argv=None):
    """Завантажити перелік з GitHub та зберегти підтримувані зображення.

    Args:
        argv (list[str] | None): Аргументи командного рядка; None читає sys.argv.

    Returns:
        None: Функція не повертає значення.
    """
    cfg = load_config(parse_args(argv).config)
    with urllib.request.urlopen(cfg.downloads.source_url, timeout=cfg.downloads.listing_timeout) as response:
        files = json.load(response)

    allowed = {ext.lower() for ext in cfg.paths.extensions}
    for item in files:
        name = Path(item['name']).name
        if item['type'] == 'file' and Path(name).suffix.lower() in allowed:
            download(item['download_url'], cfg.slides_dir / name, cfg.downloads.file_timeout)


if __name__ == '__main__':
    main()

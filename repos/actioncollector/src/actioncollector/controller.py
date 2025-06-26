from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path


def get_config_path():
    """Get the path to the config file."""
    config_dir = Path.home() / ".actioncollector"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.json"


def load_config():
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save configuration to file."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def get_bundled_assets_path() -> str:
    """Get path to bundled service account credentials"""
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        credentials_path = bundle_dir / "assets"
        if credentials_path.exists():
            return str(credentials_path)

    return "assets"


if __name__ == "__main__":
    multiprocessing.freeze_support()

    from gooey import Gooey, GooeyParser

    from actioncollector.main import run

    assets_path = get_bundled_assets_path()

    @Gooey(program_name="Action Collector", image_dir=assets_path)
    def main():
        config = load_config()
        parser = GooeyParser(description="Action Collector")
        parser.add_argument(
            "--username",
            help="Username for action collection",
            default=config.get("username", None),
        )
        args = parser.parse_args()

        if args is None:
            return

        if args.username:
            config["username"] = args.username
            save_config(config)

        run(args.username)

    main()

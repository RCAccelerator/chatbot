"""Main package entrypoint"""
from pathlib import Path
import sys
import subprocess

def main():
    """Main entrypoint for the chatbot"""
    package_dir = Path(__file__).parent
    data_dir = package_dir.joinpath("data").as_posix()
    app_path = package_dir.joinpath("app.py").as_posix()

    subprocess.run(["chainlit", "run", app_path, *sys.argv[1:]],
                   cwd=data_dir, check=True)

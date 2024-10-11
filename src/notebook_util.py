import os
from pathlib import Path

# notebook 파일 기준 경로로 쓰게 해줌
def nb_path(path):
    current_dir = Path().resolve() 
    absolute_path = current_dir / path
    return absolute_path
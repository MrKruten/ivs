import os


def create_file_inside_of_dir(dirname: str, filename: str):
    os.makedirs(dirname, exist_ok=True)
    open(os.path.join(dirname, filename), 'a')
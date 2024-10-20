import os

def preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_json_file_paths(folder_path: str) -> list[str]:
    """
    Get the json paths within folder

    :param folder_path: path of the folder to search
    :return: a list of all json file paths within the folder
    """
    if not folder_path.endswith('/'):
        folder_path += '/'
    all_files = os.listdir(folder_path)
    return [f'{folder_path}{f}' for f in all_files if f.endswith('.json')]

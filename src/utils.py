import os
import unicodedata
import emoji


def preprocess(text: str) -> str:
    """
    preprocess the raw tweet before feeding into the model

    :param text: input text to process
    :return: cleaned text
    """
    text = text.replace('\n', ' ')
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        # remove accents (e.g., é -> e)
        t = unicodedata.normalize('NFKD', t).encode(
            encoding='ascii',
            errors='ignore'
        ).decode('utf-8')

        new_text.append(t)

    processed_text = " ".join(new_text)
    return remove_emojis(processed_text)


def remove_emojis(text: str) -> str:
    """
    remove emojis in the text

    :param text: input text str
    :return: cleaned text
    """
    return emoji.replace_emoji(text, replace='')


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

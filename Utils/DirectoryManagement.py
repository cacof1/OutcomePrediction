import os
import re


def create_unique_folder(base_path):
    """
    Creates a folder at base_path.
    If it exists, appends _1, _2, ... until a unique name is found.
    Returns the path of the created folder.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return base_path

    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)

    # Regex to match folder names like "name", "name_1", "name_2", etc.
    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")

    existing_folders = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and pattern.match(name)
    ]

    # Find the highest suffix
    max_suffix = 0
    for folder in existing_folders:
        match = pattern.match(folder)
        if match:
            num = match.group(1)
            if num:
                max_suffix = max(max_suffix, int(num))
            else:
                max_suffix = max(max_suffix, 0)

    new_folder_name = f"{base_name}_{max_suffix + 1}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

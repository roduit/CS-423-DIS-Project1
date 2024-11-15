# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-07-02 -*-
# -*- Last revision: 2024-09-30 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Function to automatically change the revision date-*-

import datetime
import os
import subprocess
import re


def get_modified_files() -> list:
    """Get the list of modified files in the current git repository.

    Returns:
        list: A list of modified files.
    """
    result = subprocess.run(["git", "diff", "--name-only", "--cached"], stdout=subprocess.PIPE)
    files = result.stdout.decode("utf-8").split()
    return [f for f in files if f.endswith(".py")]


def get_git_username() -> str:
    """Get the GitHub username from the git config.

    Returns:
        str: The GitHub username.
    """
    result = subprocess.run(["git", "config", "user.name"], stdout=subprocess.PIPE)
    username = result.stdout.decode("utf-8").strip()
    return username


def update_revision_date(file_path: str, username: str):
    """Update the last revision date in the file.

    Args:
        file_path (str): The path to the file.
        username (str): The GitHub username.

    Returns:
        None
    """
    with open(file_path, "r") as file:
        content = file.readlines()

    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Update the last revision date line
    for i, line in enumerate(content):
        if line.startswith("# -*- Last revision:"):
            content[i] = f"# -*- Last revision: {current_date} by {username} -*-\n"
            break

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.writelines(content)


def main():
    """Main function to update the revision date in all modified files."""
    modified_files = get_modified_files()
    username = get_git_username()
    for file in modified_files:
        update_revision_date(file, username)
        subprocess.run(["git", "add", file])


if __name__ == "__main__":
    main()

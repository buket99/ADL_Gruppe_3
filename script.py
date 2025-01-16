import os
from pathlib import Path

def generate_tree(folder_path, skip_folders, output_file):
    """
    Generates a tree structure of the folder, excluding specified folders and invisible files.
    """
    tree_output = []
    for root, dirs, files in os.walk(folder_path):
        # Skip specified folders
        dirs[:] = [d for d in dirs if d not in skip_folders and not d.startswith('.')]
        files = [f for f in files if not f.startswith('.')]  # Skip invisible files
        level = root.replace(folder_path, "").count(os.sep)
        indent = " " * 4 * level
        tree_output.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            tree_output.append(f"{sub_indent}{file}")
    return "\n".join(tree_output)

def ensure_directory_exists(file_path):
    """
    Ensures that the directory for the given file path exists.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_and_write(folder_path, output_file, skip_folders, skip_files):
    """
    Parses all files in a folder and writes their content into a .txt file
    with headers indicating the corresponding file.
    """
    # Ensure the output directory exists
    ensure_directory_exists(output_file)

    # Generate tree overview and write to file
    tree_structure = generate_tree(folder_path, skip_folders, output_file)
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("PROJECT TREE OVERVIEW\n")
        outfile.write("=====================\n")
        outfile.write(tree_structure)
        outfile.write("\n\n")

        # Walk through files and write their content
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in skip_folders and not d.startswith('.')]
            for file in files:
                if file in skip_files or file.startswith('.') or not file.endswith((".html", ".py", ".md", ".qmd")):
                    continue
                file_path = Path(root) / file
                outfile.write(f"==== {file_path} ====\n")  # Header for each file
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # Separate contents with a blank line
                except UnicodeDecodeError:
                    print(f"Skipped file due to encoding error: {file_path}")


folder_to_parse = "/Users/moki/University/test/ADL_Gruppe_3"
output_file_path = "/Users/moki/University/test/ADL_Gruppe_3/parsed_files.txt"
skip_folders = [".quarto", "_book"]  # Folders to skip
skip_files = [os.path.basename(__file__)]  # Skip this script

parse_and_write(folder_to_parse, output_file_path, skip_folders, skip_files)
import os
import yaml

# Updated folders to skip
SKIP_FOLDERS = {
    "__pycache__", ".git", ".vscode", ".idea", ".DS_Store", "node_modules",
    "venv", "env", "dist", "build", ".pytest_cache", ".mypy_cache", ".coverage",
    ".gitignore", ".history", "coverage", "logs", "tmp", ".next", ".nuxt", ".expo",
    ".venv", "model", "deepseek-env"
}

def scan_project(root_dir, output_yaml="project_structure.yaml"):
    project_structure = {}

    for root, dirs, files in os.walk(root_dir):
        # Normalize path and exclude skipped folders
        rel_path = os.path.relpath(root, root_dir)
        if any(skip in rel_path.split(os.sep) for skip in SKIP_FOLDERS):
            continue

        project_structure[rel_path] = files

    with open(output_yaml, "w") as yaml_file:
        yaml.dump(project_structure, yaml_file, default_flow_style=False)

    print(f"YAML file '{output_yaml}' created successfully!")

# Run script on the current directory
if __name__ == "__main__":
    scan_project(os.getcwd())

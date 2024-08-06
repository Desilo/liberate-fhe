import pathlib
import shutil


def clean_project():
    """
        Remove common build directories and *.so files in a Python project.
    """
    # Directories to remove
    dirs_to_remove = [
        "__pycache__", ".pytest_cache",
        "build", "*.egg-info",  # "dist",
    ]
    # File patterns to remove
    files_to_remove = ["*.so"]

    for path in pathlib.Path(".").rglob("*"):
        # Remove specified directories
        if path.is_dir() and any(path.match(d) for d in dirs_to_remove):
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed directory: {path}")
        # Remove specified files
        elif path.is_file() and any(path.match(f) for f in files_to_remove):
            path.unlink()
            print(f"Removed file: {path}")


if __name__ == "__main__":
    clean_project()

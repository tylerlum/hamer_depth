import os

def get_parent_folder_of_package(package_name: str) -> str:
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = package.__file__
    if package_path is None:
        raise ValueError(f"Package {package_name} does not have a valid __file__ attribute")
    package_path = os.path.abspath(package_path)

    # Get the parent directory of the package directory
    return os.path.dirname(os.path.dirname(package_path))


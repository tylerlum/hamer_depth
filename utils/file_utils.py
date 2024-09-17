import os

def get_parent_folder_of_package(package_name):
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = os.path.abspath(package.__file__)

    # Get the parent directory of the package directory
    return os.path.dirname(os.path.dirname(package_path))


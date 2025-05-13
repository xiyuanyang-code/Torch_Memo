import importlib
import pkg_resources
import sys
import os


def check_package_(package_name, required_version=None):
    """
    Check if a Python package is installed correctly.

    Parameters:
        package_name (str): Name of the package.
        required_version (str, optional): Required version, e.g., '>=1.0.0'.

    Returns:
        dict: A dictionary containing the check results.
    """
    result = {
        "package": package_name,
        "installed": False,
        "version": None,
        "version_ok": None,
        "importable": False,
        "error": None,
    }

    try:
        # Check if the package is installed and its version
        installed_dist = pkg_resources.get_distribution(package_name)
        result["installed"] = True
        result["version"] = installed_dist.version

        if required_version:
            # Check if the version meets the requirement
            version_ok = pkg_resources.parse_version(
                installed_dist.version
            ) >= pkg_resources.parse_version(required_version.lstrip(">=<>~!"))
            result["version_ok"] = version_ok

        # Try to import the package
        importlib.import_module(package_name)
        result["importable"] = True

    except pkg_resources.DistributionNotFound:
        result["error"] = f"{package_name} is not installed"
    except pkg_resources.VersionConflict as e:
        result["error"] = f"Version conflict: {e}"
    except ImportError as e:
        result["error"] = f"Import error: {e}"
    except Exception as e:
        result["error"] = f"Unknown error: {e}"

    return result


def Check1_Package(required_packages):
    """Check 1 for packages

    Args:
        required_packages (dict): For all the packages names and versions required to be installed

    Returns:
        bool: whether the check is passed
    """
    print("Test 1: Check Required Packages...")
    print("=" * 50)
    print("Python Package Installation Check")
    print(f"Python Version: {sys.version}")
    print("=" * 50)

    all_ok = True
    for package, version in required_packages.items():
        result = check_package_(package, version)

        print(f"\nChecking: {package}")
        print(f"Installed: {'Yes' if result['installed'] else 'No'}")

        if result["installed"]:
            print(f"Version: {result['version']}")
            if version:
                print(f"Version Requirement: {version}")
                print(
                    f"Version Meets Requirement: {'Yes' if result['version_ok'] else 'No'}"
                )
            print(f"Importable: {'Yes' if result['importable'] else 'No'}")

        if result["error"]:
            print(f"Error: {result['error']}")
            all_ok = False

        if (
            result["installed"]
            and (version is None or result["version_ok"])
            and result["importable"]
        ):
            print("Status: ✔️ Passed")
        else:
            print("Status: ❌ Failed")
            all_ok = False

    print("\n" + "=" * 50)
    print(
        f"Overall Check Result: {'All packages passed the check' if all_ok else 'Some packages have issues'}"
    )
    print("=" * 50)

    return all_ok


def Check2_FileStructure():
    """Check whether several folders are created in the current directory

    Returns:
        bool: whether the check is passed.
    """
    print("Test 2: Check existing Files...")
    folders = ["data", "img", "log"]
    status = True

    for folder in folders:
        folder_path = os.path.join(os.getcwd(), folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"{folder_path} check passed.")
        else:
            print(f"Error! Creating {folder_path} failed.")
            status = False
            break
    return status


def main():
    print("Hello world! Welcome to the world of Deep Learning!")
    print("This Python script ensures that several packages are installed correctly.")

    # Define the packages to check and their version requirements
    # TODO Finish requirements.txt
    required_packages = {
        "numpy": ">=1.18.0",
    }

    assert Check1_Package(required_packages=required_packages)
    assert Check2_FileStructure()


if __name__ == "__main__":
    main()
    # !Attention, time for check1 is too long, considering banning check1

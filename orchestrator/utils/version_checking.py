import warnings
import urllib.request
import json
from importlib.metadata import version as get_version
from packaging.version import parse as parse_version


def _check_latest_version(package_name: str):
    """
    This function compares the installed version of the specified package with
    the latest version available on PyPI. If the installed version is older,
    it issues a warning.

    :param package_name: The name of the package (must match PyPI name).
    :type package_name: str
    :returns: None
    :rtype: None
    """
    try:
        current_version = get_version(package_name)
        with urllib.request.urlopen(
                f"https://pypi.org/pypi/{package_name}/json",
                timeout=5) as response:
            data = json.load(response)
            latest_version = data["info"]["version"]
            if parse_version(latest_version) > parse_version(current_version):
                warnings.warn(
                    f'''A newer version of '{package_name}' is available:
                    {latest_version} (current: {current_version})''',
                    UserWarning)
    except Exception as e:
        print(e)

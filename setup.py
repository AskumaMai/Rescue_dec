from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent

def read_req():
	req = root / "requirements.txt"
	if not req.exists():
		return []
	lines = [l.strip() for l in req.read_text(encoding='utf8').splitlines()]
	return [l for l in lines if l and not l.startswith('#')]

setup(
	name="rescue_cli",
	version="0.1.0",
	description="Rescue CLI and tools",
	long_description=(root / "README.md").read_text(encoding='utf8') if (root / "README.md").exists() else "",
	long_description_content_type="text/markdown",
	author="",
	packages=find_packages(exclude=("tests",)),
	include_package_data=True,
	install_requires=read_req(),
	entry_points={
		"console_scripts": [
			"rescue=rescue.__main__:cli",
		]
	},
	python_requires=">=3.8",
)

python3 -m pip install --upgrade pip

rm -rf dist build *.egg-info

python -m build

python3 -m pip install --upgrade twine

python3 -m twine upload --repository pypi dist/*
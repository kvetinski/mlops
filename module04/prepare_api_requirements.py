"""Pin the serving-only dependencies to versions installed in the active venv.

Run after the local API successfully loads your packaged model.
"""
from importlib.metadata import version
from pathlib import Path

root = Path(__file__).resolve().parent
mlflow_requirements = root / 'artifacts' / 'model' / 'requirements.txt'
if not mlflow_requirements.is_file():
    raise SystemExit(
        f'Missing MLflow model requirements: {mlflow_requirements}\n'
        'First run: python module04/package_model.py'
    )

packages = ('fastapi', 'uvicorn', 'skops')
lines = ['# Serving dependencies pinned from the working local environment.']
for package in packages:
    lines.append(f'{package}=={version(package)}')

output = root / 'requirements-api.txt'
output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(output)
print(output.read_text(), end='')

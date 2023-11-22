echo 'isort:'
poetry run isort .
echo 'flake8:'
poetry run flake8 .
echo 'black:'
poetry run black .

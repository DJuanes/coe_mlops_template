# CoE MLOps Template

## Virtual environment

```bash
python3 -m venv venv
source ./venv/scripts/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir coe_template --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

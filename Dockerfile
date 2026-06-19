FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.4" "pyarrow>=15" \
    "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.6" \
    "streamlit>=1.33" "pydeck>=0.9"

COPY src ./src
COPY data ./data
COPY .streamlit ./.streamlit

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

CMD ["uvicorn", "parksight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

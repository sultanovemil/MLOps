FROM python:3.10

EXPOSE 8501

USER root

WORKDIR /app

COPY scripts /app/scripts/

COPY main.py /app/

COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y python3-pip python3-venv

RUN python3 -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD streamlit run main.py

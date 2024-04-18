FROM jenkins/jenkins:lts

USER root

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv

RUN python3 -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install pandas && \
    pip install scikit-learn

USER jenkins

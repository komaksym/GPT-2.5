FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

WORKDIR /code

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Then install packages in the venv
COPY ./serving/uv.lock /code/uv.lock
COPY ./serving/pyproject.toml /code/pyproject.toml

RUN uv sync --frozen

COPY ./serving/app /code/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

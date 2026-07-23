ARG BASE_IMAGE=conf-summarizer:jetson-orin
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN PIP_VERBOSE=0 python3 -m pip install --no-deps --no-build-isolation -e /app

# The base venv's interpreter symlink points under root's private home.  Copy
# that small runtime to /opt and repoint the venv so the container can run as
# the host UID without opening traversal access to /root.
RUN cp -aL \
    /root/.local/share/uv/python/cpython-3.10-linux-aarch64-gnu \
    /opt/python-runtime
RUN ln -sfn /opt/python-runtime/bin/python3.10 /opt/venv/bin/python

ENTRYPOINT ["python3", "-m", "local_video_editor"]
CMD ["--help"]

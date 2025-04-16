FROM registry.access.redhat.com/ubi9/python-312

USER root
RUN groupadd -g 65532 chatgroup && \
    useradd -u 65532 -g chatgroup chatuser

WORKDIR /app

COPY src/ .
COPY pdm.lock pyproject.toml Makefile .
RUN make install-pdm install-global
RUN chown -R chatuser:chatgroup /app

USER chatuser
EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host",  "0.0.0.0"]

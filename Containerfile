FROM registry.access.redhat.com/ubi9/python-312

USER root
RUN groupadd -g 65532 chatgroup && \
    useradd -u 65532 -g chatgroup chatuser

WORKDIR /app

COPY uv.lock pyproject.toml Makefile LICENSE README.md .
COPY src/ src/
RUN make install-uv install-global

RUN chown -R chatuser:chatgroup /app

USER chatuser
EXPOSE 8000

RUN cp -r src/rca_accelerator_chatbot/data/* . && \
    cp -r src/rca_accelerator_chatbot/data/.chainlit .

CMD ["chainlit", "run", "src/rca_accelerator_chatbot/app.py", "--host",  "0.0.0.0"]

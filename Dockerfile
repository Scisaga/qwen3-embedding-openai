FROM vllm/vllm-openai:v0.9.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY embedding_service.py /app/embedding_service.py
COPY mcp_server.py /app/mcp_server.py
COPY server.py /app/server.py
COPY static /app/static

EXPOSE 12301
ENTRYPOINT ["python", "-u", "server.py"]

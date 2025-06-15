FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt update && apt install -y python3-pip git
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

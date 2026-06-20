FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from make_dataset import make_dataset; make_dataset()"

EXPOSE 7860

CMD ["python", "app.py"]

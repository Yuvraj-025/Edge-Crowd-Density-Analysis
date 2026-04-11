FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to cache them
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app.py .
COPY backend/wsgi.py .

# Copy the trained model
# Note: since the build context is the root directory, we can access ML_Service/crowd_counting.pth
COPY ML_Service/crowd_counting.pth crowd_counting.pth

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]

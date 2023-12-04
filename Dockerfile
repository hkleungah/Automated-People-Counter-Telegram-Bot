FROM python:3.10.13

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install cmake
RUN apt-get update && apt-get install -y cmake

RUN pip install ultralytics

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Clone the ByteTrack repository
RUN git clone https://github.com/ifzhang/ByteTrack.git /app/ByteTrack
WORKDIR /app/ByteTrack

# Install the requirements
RUN sed -i 's/onnx==1.8.0/#onnx==1.8.0/g' requirements.txt \
    && sed -i 's/onnxruntime==1.12.0/#onnxruntime==1.12.0/g' requirements.txt \
    && cat requirements.txt

#RUN pip install --no-cache-dir -r requirements.txt 

RUN pip install numpy==1.23.5 onnx==1.14.1 onnxruntime==1.16.0
RUN pip install supervision==0.1.0 

RUN python setup.py develop
RUN pip install opencv-python-headless
RUN pip install ffmpeg-python
RUN pip install Cython && pip install -q cython_bbox && pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox && pip install -q onemetric && pip install -q loguru lap thop

RUN pip install python-telegram-bot

WORKDIR /app
COPY . /app

CMD ["python", "app.py"]
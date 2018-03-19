FROM floydhub/dl-base:2.1.0-py3.22

ARG model_bucket=http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/tf/c3d
ARG model_file=max_c3d_sports1m_tf.tar.gz

WORKDIR /workspace
RUN mkdir assets
RUN wget -nv ${model_bucket}/${model_file} --output-document=/workspace/assets/${model_file}
RUN tar -x -C assets/ -f assets/${model_file} -v

ARG tensorflow_version=1.6.0
RUN apt-get update && apt-get install ffmpeg
RUN pip install --upgrade pip && \
    pip install tensorflow==${tensorflow_version} && \
    pip install Pillow && \
    pip install flask-restplus && \
    pip install ffmpy && \
    pip install opencv-python

COPY . /workspace

EXPOSE 5000

CMD python app.py
FROM codait/max-base

ARG model_bucket=http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/tf/c3d
ARG model_file=max_c3d_sports1m_tf.tar.gz

WORKDIR /workspace
RUN wget -nv --show-progress --progress=bar:force:noscroll ${model_bucket}/${model_file} --output-document=/workspace/assets/${model_file}
RUN tar -x -C assets/ -f assets/${model_file} -v && rm assets/${model_file}

ARG tensorflow_version=1.6.0
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install tensorflow==${tensorflow_version} && \
    pip install Pillow && \
    pip install ffmpy && \
    pip install opencv-python

COPY . /workspace

EXPOSE 5000

CMD python app.py
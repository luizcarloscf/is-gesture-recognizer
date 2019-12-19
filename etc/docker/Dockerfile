FROM tensorflow/tensorflow:1.11.0-gpu-py3

RUN pip uninstall -y jupyter ipykernel                               \
    && rm -rf /notebooks                                                \ 
    && rm -f /run_jupyter.sh                                            \
    && rm -f /root/.jupyter/jupyter_notebook_config.py                  \
    && rm -f /tensorflow_gpu-1.11.0-cp35-cp35m-manylinux1_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libsm6 libxext6 libxrender-dev 


WORKDIR /opt/is

COPY src/ src
COPY etc/model/* src/is_gesture_recognizier/

COPY requirements.txt  /requirements.txt
COPY etc/conf/options.json .

RUN pip3 install -r /requirements.txt

# ENTRYPOINT ["python3", "/src/is_gesture_recognizier/main.py"]
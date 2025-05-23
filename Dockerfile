FROM kubeflownotebookswg/jupyter-pytorch-full:latest
USER root
WORKDIR /workspace
COPY . .
RUN pip install "numpy<=1.23.5"
ENTRYPOINT []
CMD  ["python", "train.py"]
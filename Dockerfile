FROM kubeflownotebookswg/jupyter-pytorch-full:latest
USER root
RUN mkdir -p /workspace && chmod +777 /workspace 
USER jovyan
WORKDIR /workspace
COPY . .
RUN pip install "numpy<=1.23.5"
ENTRYPOINT ""
CMD  ["python", "train.py"]
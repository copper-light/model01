FROM kubeflownotebookswg/jupyter-pytorch-full:latest
WORKDIR /workspace
COPY . .
USER root
RUN chmod +777 /workspace 
USER jovyan
RUN pip install "numpy<=1.23.5"
ENTRYPOINT []
CMD  ["python", "train.py"]
FROM kubeflownotebookswg/jupyter-pytorch-full:latest
RUN mkdir -p /workspace &&  chown jovyan /workspace
USER jovyan
WORKDIR /workspace
COPY . .
RUN pip install "numpy<=1.23.5"
ENTRYPOINT ""
CMD  ["python", "train.py"]
FROM kubeflownotebookswg/jupyter-pytorch-full:latest
WORKDIR /workspace
COPY . .
RUN pip install "numpy<=1.23.5" && pip install -e .
ENTRYPOINT ""
CMD  ["python", "train.py"]
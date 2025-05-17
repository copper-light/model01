FROM kubeflownotebookswg/jupyter-pytorch-full:latest
WORKDIR /workspace
RUN pip install "numpy<=1.23.5" && pip install -e .

COPY . .
ENTRYPOINT ""
CMD  ["python", "train.py"]
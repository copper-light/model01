FROM kubeflownotebookswg/jupyter-pytorch-full:latest
WORKDIR /workspace
RUN pip install "numpy<=1.23.5"

COPY . .
CMD ["python", "train.py"]
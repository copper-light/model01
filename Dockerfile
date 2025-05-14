FROM kubeflownotebookswg/jupyter-pytorch-full:latest
WORKDIR /workspace
RUN pip install "numpy<1.25.2"

COPY . .
CMD ["python", "train.py"]
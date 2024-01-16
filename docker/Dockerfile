# Start from the PyTorch container
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update 
RUN apt-get install -y git


# Clone the repository
WORKDIR /workspace
RUN git clone https://github.com/utkutpcgl/ViTFreeze.git
# Change to the directory containing requirements.txt
WORKDIR /workspace/ViTFreeze/docker
RUN pip install -r /workspace/ViTFreeze/docker/requirements.txt
WORKDIR /workspace/

# Apply modifications to timm package
RUN cp ViTFreeze/files_to_replace/transforms.py /opt/conda/lib/python3.10/site-packages/timm/data/transforms.py
RUN cp ViTFreeze/files_to_replace/vision_transformer.py /opt/conda/lib/python3.10/site-packages/timm/models/vision_transformer.py

# Change to the final working directory
WORKDIR /workspace/ViTFreeze
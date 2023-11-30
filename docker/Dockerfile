# Start from the PyTorch container
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update 
RUN apt-get install -y git


# Clone the repository
WORKDIR /workspace
RUN git clone https://github.com/utkutpcgl/ViTFreeze.git
RUN pip install -r requirements.txt

# Apply modifications to timm package
RUN cp freezeout_localmim_rho/files_to_replace/transforms.py /opt/conda/lib/python3.10/site-packages/timm/data/transforms.py
RUN cp freezeout_localmim_rho/files_to_replace/vision_transformer.py /opt/conda/lib/python3.10/site-packages/timm/models/vision_transformer.py

# Change to the final working directory
WORKDIR /workspace/freezeout_localmim_rho
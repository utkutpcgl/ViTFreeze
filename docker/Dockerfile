# Start from the PyTorch container
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update 
RUN apt-get install -y git
# Install timm
RUN pip install timm==0.9.7


# Clone the repository
WORKDIR /workspace
RUN git clone https://github.com/utkutpcgl/freezeout_localmim_rho.git

# Apply modifications to timm package
COPY freezeout_localmim_rho/files_to_replace/transforms.py /opt/conda/lib/python3.10/site-packages/timm/data/transforms.py
COPY freezeout_localmim_rho/files_to_replace/vision_transformer.py /opt/conda/lib/python3.10/site-packages/timm/models/vision_transformer.py

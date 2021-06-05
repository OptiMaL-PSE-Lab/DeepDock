# The base image we are going to use.
FROM nvcr.io/nvidia/pytorch:19.10-py3

#Do some basic preparations
RUN conda install -c anaconda joblib -y && \
    conda install -c conda-forge tensorflow -y

RUN pip install ipython jupyter jupyter-tensorboard --upgrade && \
    jupyter tensorboard enable --system

#RUN pip install packages
RUN conda install -c conda-forge rdkit=2019.09.1 -y && \
    pip install \
        cupy-cuda101 \
        torch==1.4.0 \
        torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html \
        torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html \
        torch-cluster==1.5.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html \
        torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html \
        torch_geometric==1.4.3 \
        numpy==1.16.6 \
        plyfile==0.7.2 \
        pandas==0.25.1 \
        networkx==2.5 \
        scikit_learn==0.21.3 \
        matplotlib==3.0.2 \
        transformers==4.2.2 \
        trimesh==3.6.5 \
        Biopython \
        py3Dmol

WORKDIR /
RUN wget --no-check-certificate https://github.com/PyMesh/PyMesh/releases/download/v0.2.0/pymesh2-0.2.0-cp36-cp36m-linux_x86_64.whl && \
    pip install pymesh2-0.2.0-cp36-cp36m-linux_x86_64.whl && \
    git clone https://github.com/shenwanxiang/ChemBench.git && \
    cd ChemBench && \
    pip install -e .

    
# Install APBS, PDB2PQR and MSMSto calculate target mesh with MaSIF
# install necessary dependencies
RUN apt-get update && \
    apt-get install -y wget git unzip cmake vim libgl1-mesa-glx

# DOWNLOAD/INSTALL APBS
RUN mkdir /install
WORKDIR /install
RUN git clone https://github.com/Electrostatics/apbs-pdb2pqr && \
    git clone https://github.com/swig/swig.git

WORKDIR /install/swig
RUN git checkout tags/v4.0.2 && \
    apt-get install -y automake && \
    ./autogen.sh && \
    ./configure && \
    apt-get install -y bison flex && \
    make && \
    make install

WORKDIR /install/apbs-pdb2pqr
RUN git checkout b3bfeec && \
    git submodule update --init --recursive && \
    cmake -DGET_MSMS=ON apbs && \
    make && \
    make install && \
    cp -r /install/apbs-pdb2pqr/apbs/externals/mesh_routines/msms/msms_i86_64Linux2_2.6.1 /root/msms/ && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# INSTALL PDB2PQR
WORKDIR /install/apbs-pdb2pqr/pdb2pqr
RUN apt-get -y install python-dev python3-dev && \
    python2.7 scons/scons.py install PREFIX="/usr/local/bin/pdb2pqr"

# Setup environment variables
ENV MSMS_BIN /usr/local/bin/msms
ENV APBS_BIN /usr/local/bin/apbs
ENV MULTIVALUE_BIN /usr/local/share/apbs/tools/bin/multivalue
ENV PDB2PQR_BIN /usr/local/bin/pdb2pqr/pdb2pqr.py

# DOWNLOAD reduce (for protonation)
WORKDIR /install
RUN ["wget", "-O", "reduce.gz", "http://kinemage.biochem.duke.edu/php/downlode-3.php?filename=/../downloads/software/reduce31/reduce.3.23.130521.linuxi386.gz"]
RUN gunzip reduce.gz && \
    chmod 755 reduce && \
    cp reduce /usr/local/bin/

# Clone deepdock and install 
WORKDIR /
RUN git clone https://github.com/OptiMaL-PSE-Lab/DeepDock.git && \
    cd deepdock && \
    git submodule update --init --recursive && \
    pip install -e . && \
    cd data && \
    wget https://ndownloader.figshare.com/files/27800817 -O dataset_CASF-2016_285.tar

# create a conda environment to run python2.7
RUN conda create -y -n python2_env \
        python=2.7 \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        jinja2  && \
    conda init bash

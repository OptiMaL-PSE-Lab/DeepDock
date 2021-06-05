
# DeepDock
Code related to: [O. Mendez-Lucio, M. Ahmad, E.A. del Rio-Chanona, J.K. Wegner,  A Geometric Deep Learning Approach to Predict Binding Conformations of Bioactive Molecules](https://doi.org/10.26434/chemrxiv.14453106.v1)

https://user-images.githubusercontent.com/48085126/116097409-68553d80-a6aa-11eb-9426-91713394c3c3.mp4

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#data">Data</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This method is based on geometric deep learning and is capable of predicting the binding conformations of ligands to protein targets. Concretely, the model learns a statistical potential based on distance likelihood which is tailor-made for each ligand-target pair. This potential can be coupled with global optimization algorithms to reproduce experimental binding conformations of ligands.

We showed that:
*  Geometric deep learning can learn a potential based on distance likelihood for ligand-target interactions 
*  This potential performs similar or better than well-established scoring functions for docking and screening tasks
*  It can be coupled with global optimization algorithms to reproduce experimental binding conformations of ligands

![Fig1](https://user-images.githubusercontent.com/48085126/116094593-f5e35e00-a6a7-11eb-871a-2ef80002b824.jpg)


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

This package runs using Pytorch and Pytorch Geometric. On top it uses standard packages such as pandas and numpy. For the complete list have a look into the [requirements.txt](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/requirements.txt) file
* install requirements.txt
  ```sh
  pip install -r requirements.txt
  ```
* install RDKIT 
  ```sh
  conda install -c conda-forge rdkit=2019.09.1
  ```
### Installation

#### Using Dockerfile
To build image and run from scratch:

1. Install [docker](https://docs.docker.com/install/)
2. Clone repo and move into project folder.
   ```sh
   git clone https://github.com/OptiMaL-PSE-Lab/DeepDock.git
   cd DeepDock
   ```
3. Build the docker image. This takes 20-30 mins to build
   ```sh
   docker build -t deepdock:latest .
   ```
4. Launch the container.
   ```sh
   docker run -it --rm --name deepdock-env deepdock:latest
   ```

#### From source

1. Clone the repo
   ```sh
   git clone https://github.com/OptiMaL-PSE-Lab/DeepDock.git
   ```
2. Move into the project folder and update submodules
   ```sh
   cd DeepDock
   git submodule update --init --recursive
   ```
3. Install prerequisite packages
   ```sh
   conda install -c conda-forge rdkit=2019.09.1
   pip install -r requirements.txt
   ```
4. Install DeepDock pacakge
   ```sh
   pip install -e .
   ```
   
## Data

You can get training and testing data following the next steps.

1. Move into the project data folder
   ```sh
   cd DeepDock/data
   ```
2. Use the following line to download the preprocessed data used to train and test the model. This will download two files, one containing PDBbind (2.3 GB) used for training and another containing CASF-2016 (32 MB) used for testing. These two files are enough to run all [examples](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/examples). 
   ```sh
   source get_deepdock_data.sh
   ```
2. In case you want to reproduce all results of the paper you will need to download the complete CASF-2016 set (~1.5 GB). You can do so with this command line from the data folder.
   ```sh
   source get_CASF_2016.sh
   ```
   
<!-- USAGE EXAMPLES -->
## Usage

Usage examples can be seen directly in the jupyter notebooks included in the repo. We added examples for:
* [Training the model](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/examples/Train_DeepDock.ipynb)
* [Score molecules](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/examples/Score_example.ipynb)
* [Predict binding conformation (docking)](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/examples/Docking_example.ipynb)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

[@omendezl](https://twitter.com/omendezlucio) and [@AntonioE89](https://twitter.com/antonioe89)

Project Link: [DeepDock](https://github.com/OptiMaL-PSE-Lab/DeepDock)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [MaSIF](https://github.com/LPDI-EPFL/masif)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)





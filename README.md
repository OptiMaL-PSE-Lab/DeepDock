
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

1. Clone the repo
   ```sh
   git clone https://github.com/OptiMaL-PSE-Lab/DeepDock.git
   ```
3. Move into the project folder
   ```sh
   cd DeepDock
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


<!-- USAGE EXAMPLES -->
## Usage

Usage examples can be seen directly in the jupyter notebooks included in the repo. We added examples for:
* [Training the model](https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/Train_DeepDock.ipynb)
* [Score molecules]()
* [Predict binding conformation (docking)]()


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





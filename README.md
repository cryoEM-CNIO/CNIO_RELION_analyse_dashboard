# CNIO_RELION_analyse_dashboard - README

This script launches a webapp to analyse in detail each RELION job in a given project. The webapp is organized in 5 different tabs:

* **Relion pipeline**: an overview of all the nodes in the project and their connectivity.
<img width="1438" alt="Screenshot 2022-07-14 at 20 15 37" src="https://user-images.githubusercontent.com/60991432/179053765-b0be299c-430d-4b7a-ac20-a4ececf9f49f.png">

* **Analyse micrographs**: an interactive scatter plot to analyze any micrograph's parameter vs any other (especially useful in combination with https://github.com/cryoEM-CNIO/CNIO_Ctf_Ice_Thickness!).
<img width="1438" alt="Screenshot 2022-07-14 at 20 18 51" src="https://user-images.githubusercontent.com/60991432/179054307-012e224b-678a-473d-8c2e-a45916283e81.png">

* **Analyse particles**: an interactive scatter plot to analyze any set of particle's parameter vs any other.
<img width="1438" alt="Screenshot 2022-07-14 at 20 23 23" src="https://user-images.githubusercontent.com/60991432/179055092-08e31056-253b-4627-b4f1-0cec84d2f458.png">

* **Analyse 2D Classification**: a tab dedicated to analyse or follow in real time your 2D classifications, monitoring particle distribution per class and convergence.
<img width="1438" alt="Screenshot 2022-07-14 at 20 25 21" src="https://user-images.githubusercontent.com/60991432/179055437-d23cc833-016a-4820-858a-53aac1925df7.png">

* **Analyse 3D Classification**: follow the convergence of the classification, per-class distribution of particles and alignment in each iteration of 3D classifications.
<img width="1438" alt="Screenshot 2022-07-14 at 20 27 11" src="https://user-images.githubusercontent.com/60991432/179055781-fb41965c-dd17-4322-9c8c-118d14ba1ea4.png">

* **Analyse 3D Refine**: follow in real time the convergence, angular distribution and FSC in each iteration.
![WhatsApp Image 2022-07-14 at 8 34 12 PM](https://user-images.githubusercontent.com/60991432/179057309-fedef18d-78d6-42b4-a472-6a226cad1f70.jpeg)


Note that all plots are interactive. The ones in in the **Analyse particles** and **Analyse micrographs** tabs can also be used to select a subset of micrographs/particles graphically and export them as a .star file that can then be imported into RELION:
https://user-images.githubusercontent.com/60991432/179057807-e8910017-0c4a-45ca-bd60-c0c97046a417.mov


## Installation

### Creating conda environment (or updating your python setup)
We encourage you to create a relion_dashboard conda environment that you can use to run this script and others from our repositories (CNIO_RELION_Live_Dashboard & CNIO_RELION_Analyse_Dashboard)

```
conda create -n relion_dashboard python=3.8
conda activate relion_dashboard
pip install pandas dash=="2.3.1" starfile pathlib2 numpy glob2 pathlib argparse seaborn sklearn regex dash-cytoscape
```

### Clone CNIO_RELION_analyse_dashboard
`git clone https://github.com/cryoEM-CNIO/CNIO_RELION_analyse_dashboard`

## Usage
### Run the script from your project directory

Execute the script from your project directory
`relion_dashboard.py`

If the script is not in your PATH, use the full explicit path to the script.

It admits two arguments (quite often the default parameters are ok):
* **--port**: choose the port were you want to launch the webapp (default: 8051)
* **--host**: choose the IP address where you want the webapp to be hosted (default: localhost).

Example:
`relion_dashboard.py --host 01.010.101.010 --port 805`




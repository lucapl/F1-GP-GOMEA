# F1-GP-GOMEA
Repo for [2025 GECCO automated design competition](https://gecco-2025.sigevo.org/Competition?itemId=5103)

## Installation guide

To install Framsticks, use:
```bash
bash install_frams.sh
```
To install all python libraries you will need Python3.12 and uv:
First install pipx using pip:
```bash
pip install pipx
```
install uv using pipx:
```bash
pipx install uv
```
And use:
```bash
uv sync
```
to install all necessary python libraries and create the virtual enviroment. 

## How to run

To run the algorithm use:
```bash
python3.12 run_gomea_f1.py \
	--framslib $PATH_TO_FRAMSTICK_DLL \
	--sim_location $PATH_TO_SIMFILE_FOLDER \
	--sims $sim1 $sim2 ... \
```

## Algorithm

The algorithm implements GOMEA in a GP representation of Framsticks F1 representation. First it generates a small, random population by applying mutation n times to a simplest framsticks in F1. Then it parses them to a GP representation of F1. The algorithm build a Linkage Tree based on cooccurence of types of nodes at given positions. Based on the model it performs Gene-Pool Optimal Mixing. Every x generations it performs mutation on with some probability on the entire population in order to increase diversity, as GOMEA does not introduce it itself (like new neuron weights or connections, stick modifiers). 


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

# minf-part1
University of Edinburgh UG4 Project on Automated Identification of Connected Autonomous Vehicle Risk Scenarios.

## Setup
1. Install NuScenes devkit by running ```$ pip install nuscenes-devkit &> /dev/null``` in terminal.
2. Download a portion of the full dataset from https://www.nuscenes.org/nuscenes. 
3. Extract to ```data/sets/``` with the directory name ```nuscenes```.
4. Install requirements listed in ```requirements.txt```.

## Run
Run the UI using ```python ui.py data/sets/nuscenes [dataset version]```.

If using the mini dataset, use the command ```python ui.py data/sets/nuscenes v1.0-mini```.
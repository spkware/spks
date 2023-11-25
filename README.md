# ``spks``

Electrophysiology analysis package for large-scale electrophysiology

<code style="color : name_color">! This package is under heavy development - we'll post here when stable! </code>


*This package aims to provide transparent access to the code being used to process and analyse data; the documentation will provide references and (hopefully) a description of what the code is doing in each step.*

### Instalation

We tipically install ``spks`` in environments with other packages so kept dependencies to a minimum

#### Dependencies:
   - ``numpy``      - array handling and loading 
   - ``scipy``      - interpolation and other
   - ``matplotlib`` - plotting
   - ``torch``      - speed up some tasks
   - ``pandas``     - make tables and save files
   - ``h5py``       - save dictionaries 
   - ``natsort``    - sort files
   - ``tqdm``       - progress bars
   - ``joblib``     - multiprocess pools

#### Instalation for the brave:

   1) clone the repository ``git clone https://github.com/spkware/spks.git`` usually in a separate folder
   2) install dependencies with ``pip`` or which ever way you want
   3) go into the newly created ``spks`` folder and install with ``python setup.py develop``
   
#### Install with anaconda in a new environment:

Recommended to avoid interfering with other packages or for users new to python.

1) install the [anaconda](https://www.anaconda.com/download) python distribution
2) create a new environment from a terminal ``conda create -n spks`` and activate it ``conda activate spks``
3) install ``torch`` in the environment ``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`` and then install the dependencies ``conda install pandas h5py natsort tqdm scipy joblib jupyter matplotlib`` you can use other versions of [pytorch](https://pytorch.org/get-started/previous-versions) depending on which graphics driver is installed. A gpu is not required.
4) install ``spks`` using pip ``pip install git+https://github.com/spkware/spks.git@main``








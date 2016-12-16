# LDStreamHMMLearn

## Install
* Download Miniconda for Python 3.5 (http://conda.pydata.org/miniconda.html)
* `conda config --add channels omnia`
* `conda install pyemma-dev` to get the development version
* Download msmtools from https://github.com/alexlafleur/msmtools and unzip
* execute `python setup.py install` in the msmtools root directory
* Download or clone LDStreamHMMLearn (see Running Scripts below for options)
* execute `python setup.py install` in the LDStreamHMMLearn root directory

## Running Scripts
### From PyCharm
* Create a PyCharm project containing LDStreamHMMLearn (e.g. clone it within Pycharm)
* From Preferences > Project: LDStreamHMMLearn > Project Interpreter, select the miniconda python interpreter (3.5.2 at the time of writing)


**examplary evaluation script execution:**
```
python -c 'from evaluate_deciles import Decile_Evaluator as s; dec = s(); dec.test_evaluate_deciles_qmm_mu()' 
```

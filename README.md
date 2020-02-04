Contributors
------------

* Michał Marcińczuk <michal.marcinczuk@pwr.edu.pl>


Installation
------------

Requirements
* CUDA 10.0
* python 3.6

```bash
sudo apt-get install python3-pip python3-dev python-virtualenv
sudo pip install -U pip
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

```bash
wget 'https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin' -O resources/kgr10.plain.skipgram.dim300.neg10.bin
```

Train
--------

```bash
python poldeepspatial/core/train.py
```


Run
--------

```bash
python poldeepspatial/core/interactive.py
```

Sample output
```bash
Enter (trajector indicator landmark): leki na stoliku
label-spatial: ['leki', 'na', 'stoliku']

Enter (trajector indicator landmark): lek na chorobę
label-other: ['lek', 'na', 'chorobę']
```
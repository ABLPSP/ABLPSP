This is the code repository of experiments in A Smooth Transition Between Induction and Deduction: Fast Abductive Learning Based on Probabilistic Symbol Perception.

## Environment dependency
1. Swi-Prolog
2. Python3 with Numpy, Tensorflow and Keras
3. ZOOpt (as a submodule)
   
## Install Swipl
http://www.swi-prolog.org/build/unix.html

## Install python3
https://wiki.python.org/moin/BeginnersGuide/Download

## Install required package

```python
#install numpy tensorflow keras
pip3 install numpy
pip3 install tensorflow
pip3 install keras
pip3 install zoopt
```

## Set environment variables(Should change file path according to your situation)

```python
#cd to ABL-HED
git submodule update --init --recursive

export ABL_HOME=$PWD

cp /usr/local/lib/swipl/lib/x86_64-linux/libswipl.so $ABL_HOME/src/logic/lib/
export LD_LIBRARY_PATH=$ABL_HOME/src/logic/lib
export SWI_HOME_DIR=/usr/local/lib/swipl/

#for GPU user
export LD_LIBRARY_PATH=$ABL_HOME/src/logic/lib:/usr/local/cuda:$LD_LIBRARY_PATH
```

## Install Abductive Learning code
First change the swipl_include_dir and swipl_lib_dir in setup.py to your own SWI-Prolog path.

```python
cd src/logic/prolog
python3 setup.py install
```

## Run the code
Below is an example of running the code, where the experiment is conducted on device 0 using the hms dataset, with the knowledge base access limited to 5 times.
```python
python ABL_PSP.py --device 0 --dataset hms --budget 5
```




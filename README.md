# MJX Cuda


## Installation
`sudo apt-get install xorg-dev libglu1-mesa-dev`

In your .bashrc you'll need to include:

`export CPATH=/usr/local/cuda-12.8/targets/x86_64-linux/include${CPATH:+${CPATH}}`

Then build using:

```
mkdir build
cd build
cmake ..
make -j
```

## Usage
There are two executables generated, one for tests and another testspeed

From the build folder you can run:

`./testspeed ../data/test_data/humanoid/humanoid.xml 1000 4096`

or 

`./test kinematics ../data/test_data/pendula.xml 1`
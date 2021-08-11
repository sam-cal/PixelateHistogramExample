export base=`pwd`
export vpython=$base/venv_Py3

if [ -d "$vpython" ]; then
    echo "Activating virtual python environment"
    source $vpython/bin/activate
else
    echo "Creating virtual python environment"
    virtualenv -p python3 $vpython
    source $vpython/bin/activate
    # update some core tools
    pip install --upgrade pip
    pip install --upgrade setuptools
    # install all the software
    pip install --upgrade tensorflow
    #pip install --upgrade tensorflow-model-optimization
    pip install --upgrade  keras
    pip install --upgrade pandas
    pip install --upgrade sklearn
    pip install --upgrade pydot
    pip install --upgrade matplotlib
    pip list  # show packages installed within the virtual environment
    git clone https://github.com/sam-cal/PixelateHistogram.git
#----Tested on:
# Keras                  2.4.3
# pandas                 1.1.5
# pip                    21.0.1
# pydot                  1.4.2
# setuptools             54.2.0
# sklearn                0.0
# tensorflow             2.4.1
# matplotlib             2.8.1
fi

cd $base

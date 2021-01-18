#export PYTHONPATH=/home/covid/.local/lib/python3.6/site-packages:$PYTHONPATH
ROOT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
# export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
ROOT_DIRECTORY=$ROOT_DIRECTORY/deep_prior
export PATH=$ROOT_DIRECTORY:$PATH
export PYTHONPATH=$ROOT_DIRECTORY:$PYTHONPATH

git clone https://github.com/Physical-Intelligence/openpi.git

python3 -m venv .venv --system-site-packages
source .venv/bin/activate

cd openpi/packages/openpi-client
pip install .

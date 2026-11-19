python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install msgpack_numpy
pip install h5py
pip install meshcat
pip install transforms3d
pip install pymodbus

git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi/packages/openpi-client
pip install .
sudo apt update && sudo apt -y upgrade
sudo apt -y install build-essential git wget curl unzip tmux htop pciutils numactl net-tools

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install nvidia-driver-580 nvidia-fabricmanager-580
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
nvidia-smi

sudo apt -y install cuda-toolkit-13-0
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2404/x86_64/nvidia-machine-learning-repo-ubuntu2404_1.0.0-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu2404_1.0.0-1_amd64.deb
sudo apt update
sudo apt -y install libnccl2 libnccl-dev
dpkg -l | grep nccl

sudo apt -y update
sudo apt -y install openmpi-bin libopenmpi-dev

git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j"$(nproc)" MPI=0 CUDA_HOME=/usr/local/cuda

# Get topology
export NCCL_TOPO_DUMP_FILE=$PWD/topo_real.xml
./build/all_reduce_perf -b 8M -e 8G -f 2 -g 8

cp topo_real.xml topo_fail.xml

# normal case
unset NCCL_TOPO_FILE
./build/all_reduce_perf -b 8M -e 8G -f 2 -g 8

# Link failure topology
export NCCL_TOPO_FILE=$PWD/topo_fail.xml
./build/all_reduce_perf -b 8M -e 8G -f 2 -g 8

sudo systemctl stop nvidia-fabricmanager

sudo systemctl start nvidia-fabricmanager

# Test CUDA failure common.cu:154 'unspecified launch failure'
# Test failure all_reduce.cu:518

cat <<'EOF' >> ~/.bashrc
if [ -d /usr/lib/x86_64-linux-gnu/openmpi ]; then
  export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
  export PATH=$PATH:$MPI_HOME/bin
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$MPI_HOME/lib
  export CPATH=${CPATH:+$CPATH:}$MPI_HOME/include
  export LIBRARY_PATH=${LIBRARY_PATH:+$LIBRARY_PATH:}$MPI_HOME/lib
fi
EOF

make -j"$(nproc)" MPI=1 CUDA_HOME=/usr/local/cuda

ssh-copy-id zx@192.168.11.11
ssh-copy-id zx@192.168.11.10

mpirun -np 4 -H 192.168.11.10:2,192.168.11.11:2 --bind-to none --map-by ppr:2:node --mca oob_tcp_if_include enp75s0f1np1 --mca btl_tcp_if_include enp75s0f1np1 -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET -x NCCL_SOCKET_IFNAME=enp75s0f1np1 -x NCCL_IB_HCA=mlx5_1 -x NCCL_IB_GID_INDEX=3 -x NCCL_CROSS_NIC=1 ./build/all_reduce_perf -b 8M -e 4G -f 2 -g 1

# To be continued: Multi-rail test
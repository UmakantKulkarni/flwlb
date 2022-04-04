#!/usr/bin/bash

#Auto-method - https://askubuntu.com/a/1288405/1246619 - Dont follow this link

#Manual method:


apt-get update && apt-get -y upgrade && apt-get update

apt -y install curl wget apache2-utils default-jre default-jdk wget git vim nano make g++ net-tools iproute2 libssl-dev tcpdump jq iputils-ping apt-transport-https nghttp2-client bash-completion xauth gcc autoconf libtool pkg-config sshpass python3 python3-setuptools python3-pip qt5-default x11-apps ubuntu-drivers-common feh virtualenv

cd /mydata/
mkdir tmp
#wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
#sudo sh cuda_11.6.2_510.47.03_linux.run --toolkit --toolkitpath=/mydata/cuda --tmpdir=/mydata/tmp/ --silent --override
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run
sudo sh cuda_11.5.0_495.29.05_linux.run --toolkit --toolkitpath=/mydata/cuda --tmpdir=/mydata/tmp/ --silent --override

ubuntu-drivers devices
 
#https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
apt -y install nvidia-driver-470

reboot now -h

nvidia-smi
cat /proc/driver/nvidia/gpus/

cd /mydata && git clone https://github.com/UmakantKulkarni/scripts

#https://developer.nvidia.com/compute/cudnn/secure/8.3.3/local_installers/11.5/cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive.tar.xz
#Download on MaC and copy it to server in /mydata/
chmod +x cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive.tar.xz
sudo cp -P cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive/lib/* /mydata/cuda/lib64/
sudo cp cudnn-linux-x86_64-8.3.3.40_cuda11.5-archive/include/* /mydata/cuda/include/

#https://github.com/tensorflow/tensorflow/issues/4078#issuecomment-255129832
sudo find /usr/ -name 'libcuda.so.1'
cp /usr/lib/x86_64-linux-gnu/libcuda.so.1 /mydata/cuda/lib64/

export LD_LIBRARY_PATH="/mydata/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
source ~/.bashrc

#https://docs.bazel.build/versions/main/install-ubuntu.html
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
apt -y install bazel-4.2.1

cd /mydata/
mkdir flow_pic
cd /mydata/flow_pic
virtualenv flow_pic_ml
source flow_pic_ml/bin/activate
pip3 install matplotlib pandas scikit-learn gdown numpy

cd /mydata/flow_pic
#https://www.tensorflow.org/install/source
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.8.0
./configure
#E=Enter [no clang]
#n y n E n E E n

export HOME=/mydata/temp/
bazel --output_base=/mydata/tmp/
bazel --output_user_root=/mydata/tmp/
bazel clean
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mydata/

cd /mydata/flow_pic
source flow_pic_ml/bin/activate
pip3 install /mydata/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl

python3
import tensorflow as tf
tf.config.list_physical_devices('GPU')

cd /mydata/flow_pic/
git clone https://github.com/UmakantKulkarni/FlowPic
cd FlowPic/
git checkout uk1
git pull
cd ..
gdown --id 1gz61vnMANj-4hKNvZv1KFK9LajR91X-m
unzip FlowPic_raw_csvs.zip
mv classes_csvs classes
cd FlowPic/
./traffic_csv_converter.py 
./npzToNpyDs.py
./overlap_multiclass_reg_non_bn.py


echo "Finished running setup-node.sh"


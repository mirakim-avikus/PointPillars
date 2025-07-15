# 1. libtorch
if [ ! -d "$dirname" ]; then
    mkdir -p "$dirname"
fi

wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip
mv libtorch thirdparty

# 2. onnxruntime 
wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-gpu-1.12.1.tgz
tar -xvzf onnxruntime-linux-x64-gpu-1.12.1.tgz
mv onnxruntime-linux-x64-gpu-1.12.1 thirdparty
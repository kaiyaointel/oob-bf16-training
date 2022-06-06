set -x


pip install tensorflow==1.15.2
pip3 --no-cache-dir install --upgrade -r requirements.txt
apt-get install -y libsndfile1



if [ ! -d "wavenet" ]; then
  git clone https://github.com/bfs18/nsynth_wavenet.git
fi

cd ./nsynth_wavenet/
git reset 3fa872b --hard

cp ../parallel_wavenet.patch .
git apply parallel_wavenet.patch


# dataset
ln -s /home2/tensorflow-broad-product/oob_tf_models/dpg/Parallel_WaveNet Parallel_WaveNet
MYDIR=`pwd`
echo $MYDIR
cp ../launch_benchmark.sh .
./launch_benchmark.sh

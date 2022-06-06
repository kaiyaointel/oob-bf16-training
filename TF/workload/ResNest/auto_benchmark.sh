set -x



if [ ! -d "VAE-CF" ]; then
    git clone https://github.com/QiaoranC/tf_ResNeSt_RegNet_model.git
fi

cd tf_ResNeSt_RegNet_model
git reset 073eab7 --hard

if [ ! -d "ResNest" ]; then
    ln -s /home2/tensorflow-broad-product/oob_tf_models/oob/ResNest/ ResNest
fi

cp ../resnest.patch .
git apply resnest.patch


MYDIR=`pwd`
echo $MYDIR
cp ../launch_benchmark_resnest50.sh .
cp ../launch_benchmark_resnest101.sh .
cp ../launch_benchmark_resnest50-3d.sh .

./launch_benchmark_resnest50.sh
./launch_benchmark_resnest101.sh
./launch_benchmark_resnest50-3d.sh

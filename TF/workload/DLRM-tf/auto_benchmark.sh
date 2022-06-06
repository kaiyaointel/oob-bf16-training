set -x


pip install -r requirements.txt

MYDIR=`pwd`
echo $MYDIR
./launch_benchmark.sh

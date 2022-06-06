echo "Will now run OOB BF16 train benchmark..."
filelist=`ls`
count=0
workload_dir=${PWD}
mkdir zzzlogs
for file in $filelist
do
    count=`expr ${count} + 1`
    if [ ${count} -gt 0 ]; then
        if [ "$file" == zzzmengfei ]; then
            continue
        fi
        if [ "$file" == zzzdontwork ]; then
            continue
        fi
        if [ "$file" == zzzlogs ]; then
            continue
        fi
        if [ -d "$file" ]; then
            cd $file
            echo "Will now run benchmark on the following workload:"
            echo $file
            source ./launch_benchmark.sh 2>&1 | tee ${workload_dir}/zzzlogs/$file-train.log
            cd ${workload_dir}
        fi
    fi
done

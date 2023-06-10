echo $1
if [ $1 = run ];then
    for i in 0 1 2 3 4 5
    do
    nohup python -u hm3d_data_mp.py --rank $i --world_size 6 > rank$i.log 2>&1 &
    done
elif [ $1 = kill ];then
    ps -ef | grep 'hm3d_data_mp.py' | awk '{print $2}'     
    ps -ef | grep 'hm3d_data_mp.py' | awk '{print $2}' | xargs kill -9
elif [ $1 = compress ];then
    for i in $(seq 0 51)
    do
        echo "compress $i-th tarball"
        tar -czf data/hm3d_rgbd/train_$i.tar.gz -C data/hm3d_rgbd/ train/$i/
    done
fi
caffe_home=/data/xiaobing.wang/pingjun.li/yfji/Realtime_Multi-Person_Pose_Estimation-master/caffe_train-master/build

$caffe_home/tools/caffe train --solver=hand_solver.prototxt --weights=pose_iter_102000.caffemodel --gpu=3
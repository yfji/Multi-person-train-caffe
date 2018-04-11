caffe_home=/data/xiaobing.wang/pingjun.li/yfji/Realtime_Multi-Person_Pose_Estimation-master/caffe_train-master/build

$caffe_home/tools/caffe train --solver=fashion_solver.prototxt --weights=/data/xiaobing.wang/pingjun.li/yfji/hand_labels_synth/train_upperbody/VGG_ILSVRC_19_layers.caffemodel --gpu=3 2>&1 | tee ./output.txt
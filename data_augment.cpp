#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

#include "caffe/cpm_data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// include mask_miss
template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice > param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist()/meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  //cout<<scale_multiplier<<","<<scale_abs<<","<<scale<<endl;
  //cout<<img_src.cols<<","<<img_src.rows<<endl;
  resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
  if(mode>4 && has_mask){
    resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);
  }
  if(mode>5 && has_mask){
    resize(mask_all, mask_all, Size(), scale, scale, INTER_CUBIC);
  }

  //modify meta data
  meta.objpos *= scale;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

template<typename Dtype>
Size CPMDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta, int mode) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  //LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
  
  img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(255); //for MPI, COCO with Scalar(255);
  mask_all_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
        img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
        if(mode>4 && has_mask){
          mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
        if(mode>5 && has_mask){
          mask_all_aug.at<uchar>(i,j) = mask_all.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
      }
    }
  }

  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return Size(x_offset, y_offset);
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  bool doflip;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob());
  }
  else if(param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;
    if(mode>4 && has_mask){
      flip(mask_miss, mask_miss, 1);
    }
    if(mode>5 && has_mask){
      flip(mask_all, mask_all, 1);
    }
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode) {
  
  float degree;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  }
  else if(param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
  }
  else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }
  
  Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  if(mode >4 && has_mask){
    warpAffine(mask_miss, mask_miss, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255)); //Scalar(0) for MPI, COCO with Scalar(255);
  }
  if(mode >5 && has_mask){
    warpAffine(mask_all, mask_all, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0));
  }

  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}
// end here


template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, MetaData& meta) {
  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float scale_multiplier;
  //float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
  if(dice > param_.scale_prob()) {
    img_temp = img_src.clone();
    scale_multiplier = 1;
  }
  else {
    float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
  }
  float scale_abs = param_.target_dist()/meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
  //modify meta data
  meta.objpos *= scale;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

template<typename Dtype>
Size CPMDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, MetaData& meta) {
  float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  //LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  Point2i center = meta.objpos + Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
  
  img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
        img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
      }
    }
  }

  //modify meta data
  Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return Size(x_offset, y_offset);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::swapLeftRight(Joints& j) {
  std::vector<int> right;
  std::vector<int> left;
  
  if(np==56){
    right={3,4,5, 9,10,11,15,17};
	left= {6,7,8,12,13,14,16,18}; 
  }
  else if(np==43){
	right = {3,4,5,9,10,11}; 
    left = {6,7,8,12,13,14};  
  }
  else if(np==38){
    right ={10,12,6,7,8};
    left = {9,11,3,4,5};	
  }
  else if(np==24){
	right={2,5,7,9,12,13,15,17,19,23,24};
    left={1,4,6,8,10,11,14,16,18,21,22};
  }
  else if(np==62){
    right = {3,4,5, 9,10,11,15,17,20}; 
    left =  {6,7,8,12,13,14,16,18,19}; 
  }
  
  for(int i=0; i<right.size(); i++){    
    int ri = right[i] - 1;
    int li = left[i] - 1;
    Point2f temp = j.joints[ri];
    j.joints[ri] = j.joints[li];
    j.joints[li] = temp;
    int temp_v = j.isVisible[ri];
    j.isVisible[ri] = j.isVisible[li];
    j.isVisible[li] = temp_v;
  }
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, MetaData& meta) {
  bool doflip;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    doflip = (dice <= param_.flip_prob());
  }
  else if(param_.aug_way() == "table"){
    doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
  }
  else {
    doflip = 0;
    LOG(ERROR) << "Unhandled exception!!!!!!";
  }

  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;

    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat R){
  Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, MetaData& meta) {
  
  float degree;
  if(param_.aug_way() == "rand"){
    float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
  }
  else if(param_.aug_way() == "table"){
    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
  }
  else {
    degree = 0;
    LOG(INFO) << "Unhandled exception!!!!!!";
  }
  
  Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  Mat R = getRotationMatrix2D(center, degree, 1.0);
  Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));
  
  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}
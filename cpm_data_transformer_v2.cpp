#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
using namespace cv;
using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <time.h>

#include "caffe/cpm_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
//#include <omp.h>


namespace caffe {

int crop_cnt=0;

template<typename Dtype>
void DecodeFloats(const string& data, size_t idx, Dtype* pf, size_t len) {
  memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
}

string DecodeString(const string& data, size_t idx) {
  string result = "";
  int i = 0;
  while(data[idx+i] != 0){
    result.push_back(char(data[idx+i]));
    i++;
  }
  return result;
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::ReadMetaData(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
  // ------------------- Dataset name ----------------------
  meta.dataset = DecodeString(data, offset3);
  // ------------------- Image Dimension -------------------
  float height, width;
  DecodeFloats(data, offset3+offset1, &height, 1);
  DecodeFloats(data, offset3+offset1+4, &width, 1);
  meta.img_size = Size(width, height);
  // ----------- Validation, nop, counters -----------------
  meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
  meta.numOtherPeople = (int)data[offset3+2*offset1+1];
  meta.people_index = (int)data[offset3+2*offset1+2];
  float annolist_index;
  DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
  meta.annolist_index = (int)annolist_index;
  float write_number;
  DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
  meta.write_number = (int)write_number;
  float total_write_number;
  DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
  meta.total_write_number = (int)total_write_number;

  // count epochs according to counters
  static int cur_epoch = -1;
  if(meta.write_number == 0){
    cur_epoch++;
  }
  meta.epoch = cur_epoch;
  if(meta.write_number % 1000 == 0){
    LOG(INFO) << "dataset: " << meta.dataset <<"; img_size: " << meta.img_size
        << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number
        << "; meta.total_write_number: " << meta.total_write_number << "; meta.epoch: " << meta.epoch;
  }
  if(param_.aug_way() == "table" && !is_table_set){
    SetAugTable(meta.total_write_number);
    is_table_set = true;
  }

  // ------------------- objpos -----------------------
  DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
  DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
  //meta.objpos -= Point2f(1,1);  //Python not needed
  // ------------ scale_self, joint_self --------------
  DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
  meta.joint_self.joints.resize(np_in_lmdb);
  meta.joint_self.isVisible.resize(np_in_lmdb);
  
  //cout<<"LMDB"<<endl;
  for(int i=0; i<np_in_lmdb; i++){
    DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
    DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
    //meta.joint_self.joints[i] -= Point2f(1,1); //from matlab 1-index to c++ 0-index. Python not needed
    float isVisible;
    DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
    if (isVisible == 3){
      meta.joint_self.isVisible[i] = 3;
    }
    else{
      //meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1;    
      meta.joint_self.isVisible[i]= isVisible;
      if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
         meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image, 1 means labeled and visible on image
      }
    }
  }
  
  //others (7 lines loaded)
  meta.objpos_other.resize(meta.numOtherPeople);
  meta.scale_other.resize(meta.numOtherPeople);
  meta.joint_others.resize(meta.numOtherPeople);
  for(int p=0; p<meta.numOtherPeople; p++){
    DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
    DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
    meta.objpos_other[p] -= Point2f(1,1);
    DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);
  }
  //8 + numOtherPeople lines loaded
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.joint_others[p].joints.resize(np_in_lmdb);
    meta.joint_others[p].isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
      meta.joint_others[p].joints[i] -= Point2f(1,1);
      float isVisible;
      DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
      //meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
      meta.joint_others[p].isVisible[i]= isVisible;
      if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 ||
         meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
        meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 1 means occluded by still on image
      }
    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::SetAugTable(int numData){
  aug_degs.resize(numData);     
  aug_flips.resize(numData);  
  for(int i = 0; i < numData; i++){
    aug_degs[i].resize(param_.num_total_augs());
    aug_flips[i].resize(param_.num_total_augs());
  }
  //load table files
  char filename[100];
  sprintf(filename, "../../rotate_%d_%d.txt", param_.num_total_augs(), numData);
  ifstream rot_file(filename);
  char filename2[100];
  sprintf(filename2, "../../flip_%d_%d.txt", param_.num_total_augs(), numData);
  ifstream flip_file(filename2);

  for(int i = 0; i < numData; i++){
    for(int j = 0; j < param_.num_total_augs(); j++){
      rot_file >> aug_degs[i][j];
      flip_file >> aug_flips[i][j];
    }
  }
}


template<typename Dtype>
void CPMDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  //cout<<"joints size: "<<meta.joint_self.joints.size()<<endl;
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::TransformJoints(Joints& j) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  Joints jo = j;

  std::vector<int> Benchmark_to_ours1;
  std::vector<int> Benchmark_to_ours2;
  
  if(np==56){
	Benchmark_to_ours1 = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
	Benchmark_to_ours2 = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
  }
  else if(np==43){
	Benchmark_to_ours1 = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7};
	Benchmark_to_ours2 = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6};
  }
  else if(np==38){
	Benchmark_to_ours1 = {1,6,7,9,11,6,8,10,3,2,5,4};
    Benchmark_to_ours2 = {1,7,7,9,11,6,8,10,3,2,5,4};
  }
  else if(np==21){
    Benchmark_to_ours1 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
    Benchmark_to_ours2 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
  }
  else if(np==24){
    Benchmark_to_ours1 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Benchmark_to_ours2 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  }
  else if(np==62){
    Benchmark_to_ours1 = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4, 19,18};
    Benchmark_to_ours2 = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4, 19,18};
  }

  //visible 0,1,2 are labelled in clothes dataset
  //(2)->1: labeled and visible
  //(1)->0: labeled but occluded
  //(0)->2: unlabelled

  jo.joints.resize(np);
  jo.isVisible.resize(np);
  for(int i=0;i<np_ours;i++){
	jo.joints[i] = (j.joints[FASHION_to_ours_1[i]-1] + j.joints[FASHION_to_ours_2[i]-1]) * 0.5;
	if(j.isVisible[FASHION_to_ours_1[i]-1]==2 || j.isVisible[FASHION_to_ours_2[i]-1]==2){
	  jo.isVisible[i] = 2;
	}
	else if(j.isVisible[FASHION_to_ours_1[i]-1]==3 || j.isVisible[FASHION_to_ours_2[i]-1]==3){
	  jo.isVisible[i] = 3;
	}
	else {
	  //one is 0, the total is 0 (occluded, labelled)
	  jo.isVisible[i] = j.isVisible[FASHION_to_ours_1[i]-1] && j.isVisible[FASHION_to_ours_2[i]-1];
	}
  }
  j = jo;
}

template<typename Dtype> CPMDataTransformer<Dtype>::CPMDataTransformer(const CPMTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  LOG(INFO) << "CPMDataTransformer constructor done.";
  np_in_lmdb = param_.np_in_lmdb();
  np = param_.num_parts();
  hand=param_.hand();
  upper_body=param_.upper_body();
  upper_body_crop_prob=param_.upper_crop_prob();
  dataset=param_.dataset();
  sample_min_side=param_.sample_min_side();
  
  srand(time(0));
  if(upper_body && dataset=="COCO"){
    np_in_lmdb=17;
    np_ours=12;
    np=38;  //12+13*2
  }
  else if(dataset=="COCO"){
    np_in_lmdb=17;
    np_ours=18;
    np=56;  //18+19*2
  }
  else if(dataset=="MPI"){
    np_in_lmdb=15;
    np_ours=15;
    np=43;  //15+14*2
  }
  else if(dataset=="HAND"){
    np_in_lmdb=21;
    np_ours=21;
    np=21;
  }
  else if(dataset=="FashionAI"){
    np_in_lmdb=24;
    np_ours=24;
    np=24;
  }
  else if(dataset=="FOOTTOP"){
    np_in_lmdb=19;
    np_ours=20;
    np=62;  //20+21*2
  }
  else{
    CHECK_EQ(1,0)<<"Unknown dataset";
  }
  has_mask=false;
  if(np==38 || np==56 || np==43 || np==62)
    has_mask=true;
  is_table_set = false;
  if(has_mask){
    label_channels=2*(np+1);
    label_start=np+1;
  }
  else{
    label_channels=np+1;
    label_start=0;
  }  
  LOG(INFO) <<"Current dataset: "<<dataset<<endl;
}

template<typename Dtype> void CPMDataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data) {
  //LOG(INFO) << "Function 1 is used";
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype> void CPMDataTransformer<Dtype>::Transform(const Datum& datum, Blob<Dtype>* transformed_blob) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  Transform(datum, transformed_data);
}

template<typename Dtype> void CPMDataTransformer<Dtype>::Transform_nv(const Datum& datum, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label, int cnt) {
  //std::cout << "Function 2 is used"; std::cout.flush();
  const int datum_channels = datum.channels();
  //const int datum_height = datum.height();
  //const int datum_width = datum.width();

  const int im_channels = transformed_data->channels();
  //const int im_height = transformed_data->height();
  //const int im_width = transformed_data->width();
  const int im_num = transformed_data->num();

  const int lb_num = transformed_label->num();

  CHECK_EQ(im_num, lb_num);

  CHECK_GE(im_num, 1);

  Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
  Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
  CPUTimer timer;
  timer.Start();
  Transform_nv(datum, transformed_data_pointer, transformed_label_pointer, cnt); //call function 1
  VLOG(2) << "Transform_nv: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template<typename Dtype> void CPMDataTransformer<Dtype>::Transform_nv(const Datum& datum, Dtype* transformed_data, Dtype* transformed_label, int cnt) {
  
  //TODO: some parameter should be set in prototxt
  int clahe_tileSize = param_.clahe_tile_size();
  int clahe_clipLimit = param_.clahe_clip_limit();
  //float targetDist = 41.0/35.0;
  AugmentSelection as = {
    false,
    0.0,
    Size(),
    0,
  };
  MetaData meta;
  
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int mode = 5;

  const bool has_uint8 = data.size() > 0;
  //const bool has_mean_values = mean_values_.size() > 0;
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  CHECK_GT(datum_channels, 0);
 
  CPUTimer timer1;
  timer1.Start();
  //before any transformation, get the image from datum
  Mat img = Mat::zeros(datum_height, datum_width, CV_8UC3);
  Mat mask_all, mask_miss;
  if(mode >= 5 && has_mask){
    mask_miss = Mat::ones(datum_height, datum_width, CV_8UC1);
  }
  if(mode == 6 && has_mask){
    mask_all = Mat::zeros(datum_height, datum_width, CV_8UC1);
  }

  int offset = img.rows * img.cols;
  int dindex;
  Dtype d_element;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      Vec3b& rgb = img.at<Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }

      if(mode >= 5 && has_mask){
        dindex = 4*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        if (round(d_element/255)!=1 && round(d_element/255)!=0){
          cout << d_element << " " << round(d_element/255) << endl;
        }
        mask_miss.at<uchar>(i, j) = d_element; //round(d_element/255);
      }

      if(mode == 6 && has_mask){
        dindex = 5*offset + i*img.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        mask_all.at<uchar>(i, j) = d_element;
      }
    }
  }
  VLOG(2) << "  rgb[:] = datum: " << timer1.MicroSeconds()/1000.0 << " ms";
  timer1.Start();
  
  //color, contract
  if(param_.do_clahe())
    clahe(img, clahe_tileSize, clahe_clipLimit);
  if(param_.gray() == 1){
    cv::cvtColor(img, img, CV_BGR2GRAY);
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }
  VLOG(2) << "  color: " << timer1.MicroSeconds()/1000.0 << " ms";
  timer1.Start();

  int offset3 = 3 * offset;
  int offset1 = datum_width;
  int stride = param_.stride();
  ReadMetaData(meta, data, offset3, offset1);
  if(param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
    TransformMetaJoints(meta);

  /****crop upper body if upperbody and prob***/
  too_small=false;
  // only one person in cropped image
  // if multi-person, PAF must be used
  if(upper_body && dataset=="COCO"){
    int im_w=img.cols;
    int im_h=img.rows;
    float prob = static_cast<float>(rand())/static_cast<float>(RAND_MAX); //[0,1]
    if(prob<upper_body_crop_prob){
      int ltx=1e4,lty=1e4,rbx=0,rby=0;
      for(int i=0;i<np_ours;++i){
        cv::Point2f pt=meta.joint_self.joints[i];
        if(meta.joint_self.isVisible[i]<=1){
          ltx=min(ltx,(int)pt.x);
          lty=min(lty,(int)pt.y);
          rbx=max(rbx,(int)pt.x);
          rby=max(rby,(int)pt.y);
        }
      }
      int crop_w=max(0,rbx-ltx);
      int crop_h=max(0,rby-lty);      
      
      int start_x=max(0,ltx-crop_w/8);
      int start_y=max(0,lty-crop_h/8);
      
      if(crop_w<sample_min_side || crop_h<sample_min_side){
        /****too small, do not draw****/
        too_small=true;
        img.setTo(cv::Scalar(128,128,128));
        for(int i=0;i<np_ours;++i){
          meta.joint_self.isVisible[i] = 2;
        }
        meta.objpos.x-=start_x;
        meta.objpos.y-=start_y;
        for(int p=0;p<meta.numOtherPeople;++p)
        {
          meta.objpos_other[p].x-=start_x;
          meta.objpos_other[p].y-=start_y;
          for(int j=0;j<np_ours;++j){
            meta.joint_others[p].isVisible[j]=2;
          }
        }
      }
      else{
        crop_w=min((int)(crop_w*(1+1.0/4)),im_w-start_x);
        crop_h=min((int)(crop_h*(1+1.0/4)),im_h-start_y);
        meta.objpos.x=(ltx+rbx)/2;
        meta.objpos.y=(lty+rby)/2;  //only related with objpos     
      
        if(param_.visualize()){
          std::stringstream ss;
          ss<<"raw_"<<crop_cnt<<".jpg";
          cv::imwrite(ss.str(),img);
        }
        
        img=img(cv::Rect(start_x,start_y,crop_w,crop_h));
        if(param_.visualize()){
          std::stringstream ss;
          ss<<"crop_"<<crop_cnt<<".jpg";
          cv::imwrite(ss.str(),img);
          ++crop_cnt;
        }
        
        meta.scale_self=1.0*max(1.0*crop_w/param_.crop_size_x(),1.0*crop_h/param_.crop_size_y());
        for(int i=0;i<np_ours;++i){
          meta.joint_self.joints[i].x-=start_x;
          meta.joint_self.joints[i].y-=start_y;
        }
        meta.objpos.x-=start_x;
        meta.objpos.y-=start_y;
        for(int p=0;p<meta.numOtherPeople;++p)
        {
          Joints& jo=meta.joint_others[p];
          bool other_in=false;
          for(int j=0;j<np_ours;++j){
            cv::Point2f pt=jo.joints[j];
            if(jo.isVisible[j]<=1){
              if(pt.x>=start_x && pt.y>=start_y && pt.x<start_x+crop_w && pt.y<start_y+crop_h) {
                jo.joints[j].x-=start_x;
                jo.joints[j].y-=start_y;
                other_in=true;
              }
              else{
                jo.isVisible[j]=2;
              }
            }
          }
          //if(other_in){
          meta.objpos_other[p].x-=start_x;
          meta.objpos_other[p].y-=start_y;
          //}
        }
      }    
    }
  }

  VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()/1000.0 << " ms";
  timer1.Start();
  //visualize original
  if(0 && param_.visualize()) 
    visualize(img, meta, as);

  //Start transforming
  Mat img_aug = Mat::zeros(crop_y, crop_x, CV_8UC3);
  Mat mask_miss_aug, mask_all_aug ;
  //Mat mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
  //Mat mask_all_aug = Mat::zeros(crop_y, crop_x, CV_8UC1);
  Mat img_temp, img_temp2, img_temp3; //size determined by scale
  VLOG(2) << "   input size (" << img.cols << ", " << img.rows << ")"; 
  // We only do random transform as augmentation when training.
  if (phase_ == TRAIN) {
    as.scale = augmentation_scale(img, img_temp, mask_miss, mask_all, meta, mode);
 
    as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, mask_all, meta, mode);
 
    if(0 && param_.visualize()) 
      visualize(img_temp2, meta, as);
    as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, mask_all, mask_all_aug, meta, mode);
 
    if(0 && param_.visualize()) 
      visualize(img_temp3, meta, as);
    as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, mask_all_aug, meta, mode);
    if(param_.visualize()) 
      visualize(img_aug, meta, as);
    if (mode > 4){
      resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
    if (mode > 5){
      resize(mask_all_aug, mask_all_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
    }
  }
  else {
    img_aug = img.clone();
    as.scale = 1;
    as.crop = Size();
    as.flip = 0;
    as.degree = 0;
  }
  VLOG(2) << "  Aug: " << timer1.MicroSeconds()/1000.0 << " ms";
  timer1.Start();
  //LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height 
  //          << "); flip:" << as.flip << "; degree: " << as.degree;

  //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
  offset = img_aug.rows * img_aug.cols;
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;

  for (int i = 0; i < img_aug.rows; ++i) {
    for (int j = 0; j < img_aug.cols; ++j) {
      Vec3b& rgb = img_aug.at<Vec3b>(i, j);
      transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/256.0;
      transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/256.0;
      transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/256.0;
    }
  }
  
  // label size is image size/ stride
  /***place masks at [0,np+1), only when has_mask***/
  if (mode > 4 && has_mask){
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        for (int i = 0; i < np; i++){
          float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255; //mask_miss_aug.at<uchar>(i, j); 
          if (meta.joint_self.isVisible[i] != 3){
            transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
          }
        }  
        // background channel
        if(mode == 5){
          transformed_label[np*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255;
        }
        if(mode > 5){
          transformed_label[np*channelOffset + g_y*grid_x + g_x] = 1;
          transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = float(mask_all_aug.at<uchar>(g_y, g_x)) /255;
        }
      }
    }
  }  

  generateLabelMap(transformed_label, img_aug, meta);

  VLOG(2) << "  putGauss+genLabel: " << timer1.MicroSeconds()/1000.0 << " ms";
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1) 
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  Point2f bc = centerB - centerA;
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

  int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;

  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      float dist = std::abs(ba.x*bc.y -ba.y*bc.x);
	  
      if(dist <= thre){
      //if(judge <= 1){
        int cnt = count.at<uchar>(g_y, g_x);
        //LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
        if (cnt == 0){
          entryX[g_y*grid_x + g_x] = bc.x;
          entryY[g_y*grid_x + g_x] = bc.y;
        }
        else{
          entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
          entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
      }

    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;
  int mode = 5; // TO DO: make this as a parameter

  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = label_start; i < label_channels; i++){
        if (mode == 6 && i == label_channels)
          continue;
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }
  
  vector<int> mid_1;
  vector<int> mid_2;
  int np_offset=0;
  /****add****/
  if(np==38){
	mid_1 = {2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  9, 10};
    mid_2 = {3, 4, 5, 11, 6, 7, 8, 12, 1, 9, 10, 11, 12};
	np_offset=27;
  }
  else if(np==62){
	mid_1 = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16, 11,14};
    mid_2 = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18, 20,19};
	np_offset=43;
  }
  else if(np==56){
	mid_1 = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    mid_2 = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
	np_offset=39;
  }
  else if(np==43){
	mid_1 = {0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12};
    mid_2 = {1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13};
	np_offset=29;
  }
  if(np==21 || np==24){  //No mask, No PAF
    for (int i = 0; i < np_ours; i++){
      Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
      //if(meta.joint_self.isVisible[i] <= 1 && (center.x>0.5 || center.y>0.5)){
        putGaussianMaps(transformed_label + i*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
    }
    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = 0; i < np_ours; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[np_ours*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
      }
    }
  }
  else{
	//gaussian map
	for (int i = 0; i < np_ours; i++){
		Point2f center = meta.joint_self.joints[i];
		if(meta.joint_self.isVisible[i] <= 1){
		  putGaussianMaps(transformed_label + (i+np+np_offset)*channelOffset, center, param_.stride(), 
						  grid_x, grid_y, param_.sigma()); //self
		}
		for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
		  Point2f center = meta.joint_others[j].joints[i];
		  if(meta.joint_others[j].isVisible[i] <= 1){
			putGaussianMaps(transformed_label + (i+np+np_offset)*channelOffset, center, param_.stride(), 
							grid_x, grid_y, param_.sigma());
		  }
		}
	}
	//vec map
	for(int i=0;i<mid_1.size();i++){
		  Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
		  Joints jo = meta.joint_self;
		  if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
			//putVecPeaks
			putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
					  count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
		}

		for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
			Joints jo2 = meta.joint_others[j];
			if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
			  //putVecPeaks
			  putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
					  count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
			}
		}
	}

	//put background channel
	for (int g_y = 0; g_y < grid_y; g_y++){
	    for (int g_x = 0; g_x < grid_x; g_x++){
			float maximum = 0;
			//second background channel
			for (int i = np+np_offset; i < np+np_offset+np_ours; i++){
			  maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
			}
			transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
		}
	}  
  }
  //visualize
  if(1 && param_.visualize()){
    Mat label_map;
    for(int i = label_start; i < label_channels; i++){      
      label_map = Mat::zeros(grid_y, grid_x, CV_8UC1);
      for (int g_y = 0; g_y < grid_y; g_y++){
        for (int g_x = 0; g_x < grid_x; g_x++){
          label_map.at<uchar>(g_y,g_x) = (int)(transformed_label[i*channelOffset + g_y*grid_x + g_x]*255);
        }
      }
      resize(label_map, label_map, Size(), stride, stride, INTER_LINEAR);
      applyColorMap(label_map, label_map, COLORMAP_JET);
      addWeighted(label_map, 0.5, img_aug, 0.5, 0.0, label_map);
      
      char imagename [100];
      sprintf(imagename, "augment_%04d_label_part_%02d.jpg", meta.write_number, i);
      //LOG(INFO) << "filename is " << imagename;
      imwrite(imagename, label_map);
    }
    
  }
}

void setLabel(Mat& im, const std::string label, const Point& org) {
    int fontface = FONT_HERSHEY_SIMPLEX;
    double scale = 0.5;
    int thickness = 1;
    int baseline = 0;

    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    rectangle(im, org + Point(0, baseline), org + Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    putText(im, label, org, fontface, scale, CV_RGB(255,255,255), thickness, 20);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::visualize(Mat& img, MetaData meta, AugmentSelection as) {

  Mat img_vis = img.clone();
  static int counter = 0;

  rectangle(img_vis, meta.objpos-Point2f(3,3), meta.objpos+Point2f(3,3), CV_RGB(255,255,0), CV_FILLED);
  for(int i=0;i<np_ours;i++){
    
    if(np == 21){ // hand case
      if(i < 4)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
      else if(i < 6 || i == 12 || i == 13)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
      else if(i < 8 || i == 14 || i == 15)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
      else if(i < 10|| i == 16 || i == 17)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,0), -1);
      else if(i < 12|| i == 18 || i == 19)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
      else 
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,100,100), -1);
    }
    else if(np == 9){
      if(i==0 || i==1 || i==2 || i==6)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
      else if(i==3 || i==4 || i==5 || i==7)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
      else
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
    }
    else if(np == 14 || np == 28) {//body case
      if(i < 14){
        if(i==2 || i==3 || i==4 || i==8 || i==9 || i==10)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,0,255), -1);
        else if(i==5 || i==6 || i==7 || i==11 || i==12 || i==13)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,255,0), -1);
      }
      else if(i < 16)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(0,255,0), -1);
      else {
        if(i==17 || i==18 || i==19 || i==23 || i==24)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,0,0), -1);
        else if(i==20 || i==21 || i==22 || i==25 || i==26)
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,100,100), -1);
        else
          circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(255,200,200), -1);
      }
    }
    else {
      if(meta.joint_self.isVisible[i] <= 1)
        circle(img_vis, meta.joint_self.joints[i], 3, CV_RGB(200,200,255), -1);
    }
  }
  
  line(img_vis, meta.objpos+Point2f(-368/2,-368/2), meta.objpos+Point2f(368/2,-368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(368/2,-368/2), meta.objpos+Point2f(368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(368/2,368/2), meta.objpos+Point2f(-368/2,368/2), CV_RGB(0,255,0), 2);
  line(img_vis, meta.objpos+Point2f(-368/2,368/2), meta.objpos+Point2f(-368/2,-368/2), CV_RGB(0,255,0), 2);

  for(int p=0;p<meta.numOtherPeople;p++){
    rectangle(img_vis, meta.objpos_other[p]-Point2f(3,3), meta.objpos_other[p]+Point2f(3,3), CV_RGB(0,255,255), CV_FILLED);
    for(int i=0;i<np;i++){
      if(meta.joint_others[p].isVisible[i] <= 1)
        circle(img_vis, meta.joint_others[p].joints[i], 3, CV_RGB(0,0,255), -1);
    }
  }
  
  // draw text
  if(phase_ == TRAIN){
    std::stringstream ss;
    // ss << "Augmenting with:" << (as.flip ? "flip" : "no flip") << "; Rotate " << as.degree << " deg; scaling: " << as.scale << "; crop: " 
    //    << as.crop.height << "," << as.crop.width;
    ss << meta.dataset << " " << meta.write_number << " index:" << meta.annolist_index << "; p:" << meta.people_index 
       << "; o_scale: " << meta.scale_self;
    string str_info = ss.str();
    setLabel(img_vis, str_info, Point(0, 20));

    stringstream ss2; 
    ss2 << "mult: " << as.scale << "; rot: " << as.degree << "; flip: " << (as.flip?"true":"ori");
    str_info = ss2.str();
    setLabel(img_vis, str_info, Point(0, 40));

    rectangle(img_vis, Point(0, 0+img_vis.rows), Point(param_.crop_size_x(), param_.crop_size_y()+img_vis.rows), Scalar(255,255,255), 1);

    char imagename [100];
    sprintf(imagename, "augment_%04d_epoch_%03d_writenum_%03d.jpg", counter, meta.epoch, meta.write_number);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  else {
    string str_info = "no augmentation for testing";
    setLabel(img_vis, str_info, Point(0, 20));

    char imagename [100];
    sprintf(imagename, "augment_%04d.jpg", counter);
    //LOG(INFO) << "filename is " << imagename;
    imwrite(imagename, img_vis);
  }
  counter++;
}
template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> CPMDataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    LOG(INFO) << "Datum is encoded, so decode it into a CV mat to determine size.";
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  LOG(INFO) << "Datum is not encoded. Directly determine size.";
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> CPMDataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> CPMDataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> CPMDataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void CPMDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int CPMDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void CPMDataTransformer<Dtype>::clahe(Mat& bgr_image, int tileSize, int clipLimit) {
  Mat lab_image;
  cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  vector<Mat> lab_planes(3);
  split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  Ptr<CLAHE> clahe = createCLAHE(clipLimit, Size(tileSize, tileSize));
  //clahe->setClipLimit(4);
  Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  merge(lab_planes, lab_image);

  // convert back to RGB
  Mat image_clahe;
  cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe.clone();
}

template <typename Dtype>
void CPMDataTransformer<Dtype>::dumpEverything(Dtype* transformed_data, Dtype* transformed_label, MetaData meta){
  
  char filename[100];
  sprintf(filename, "transformed_data_%04d_%02d", meta.annolist_index, meta.people_index);
  ofstream myfile;
  myfile.open(filename);
  int data_length = param_.crop_size_y() * param_.crop_size_x() * 4;
  
  //LOG(INFO) << "before copy data: " << filename << "  " << data_length;
  for(int i = 0; i<data_length; i++){
    myfile << transformed_data[i] << " ";
  }
  //LOG(INFO) << "after copy data: " << filename << "  " << data_length;
  myfile.close();

  sprintf(filename, "transformed_label_%04d_%02d", meta.annolist_index, meta.people_index);
  myfile.open(filename);
  int label_length = param_.crop_size_y() * param_.crop_size_x() / param_.stride() / param_.stride() * (param_.num_parts()+1);
  for(int i = 0; i<label_length; i++){
    myfile << transformed_label[i] << " ";
  }
  myfile.close();
}


INSTANTIATE_CLASS(CPMDataTransformer);

}  // namespace caffe
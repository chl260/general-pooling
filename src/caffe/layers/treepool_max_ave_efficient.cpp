// Copyright 2015 Chen-Yu Lee (chl260@ucsd.edu)

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
// #include "caffe/common.hpp"
// #include "caffe/syncedmem.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void GatedMaxAveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  KSIZE_ = conv_param.kernel_size();
  STRIDE_ = conv_param.stride();
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  POOLED_HEIGHT_ = static_cast<int>(ceil(static_cast<float>(HEIGHT_ - KSIZE_) / STRIDE_)) + 1;
  POOLED_WIDTH_ = static_cast<int>(ceil(static_cast<float>(WIDTH_ - KSIZE_) / STRIDE_)) + 1;

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);

    // declare filler
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().weight_filler()));

    // Initialize weight1
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, KSIZE_, KSIZE_ ));
    weight_filler->Fill(this->blobs_[0].get());

    // Set up the bias filler     
    /*
    bias_multiplier2_.reset(new SyncedMemory((KSIZE_*KSIZE_) * sizeof(Dtype)));
    Dtype* bias_multiplier_data2 = reinterpret_cast<Dtype*>(bias_multiplier2_->mutable_cpu_data());
    for (int i = 0; i < (KSIZE_*KSIZE_); ++i) {
        bias_multiplier_data2[i] = 1.;
    }   
    */  

    bias_multiplier1_.reset(new SyncedMemory((NUM_*CHANNELS_*POOLED_HEIGHT_*POOLED_WIDTH_) * sizeof(Dtype)));
    Dtype* bias_multiplier_data1 = reinterpret_cast<Dtype*>(bias_multiplier1_->mutable_cpu_data());
    for (int i = 0; i < (NUM_*CHANNELS_*POOLED_HEIGHT_*POOLED_WIDTH_); ++i) {
        bias_multiplier_data1[i] = 1.;
    }  

  }  
}

template <typename Dtype>
void GatedMaxAveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //===========================================================================================
  //split_buffer1_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //split_buffer2_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //top_buffer1_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //top_buffer2_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //mat_buffer_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //poolsize_buffer_.Reshape(bottom[0]->num(), CHANNELS_, POOLED_HEIGHT_, POOLED_WIDTH_);
  //col_buffer_.Reshape(1, 1, KSIZE_ * KSIZE_, bottom[0]->num() * CHANNELS_ * POOLED_HEIGHT_ * POOLED_WIDTH_);
  //bottom_diff_buffer_.Reshape(1, 1, NUM_ * CHANNELS_ * HEIGHT_ * WIDTH_, KSIZE_ * KSIZE_);
  weight1_diff_buffer_.Reshape(1, 1, KSIZE_ * KSIZE_, NUM_ * CHANNELS_ * POOLED_HEIGHT_ * POOLED_WIDTH_);
  //COUNT2_ = KSIZE_ * KSIZE_ * bottom[0]->num() * CHANNELS_ * POOLED_HEIGHT_ * POOLED_WIDTH_;
  //===========================================================================================

}

template <typename Dtype>
void GatedMaxAveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "Enter CPU code ....";
}

template <typename Dtype>
void GatedMaxAveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(INFO) << "Enter CPU code ....";
}

INSTANTIATE_CLASS(GatedMaxAveLayer);
REGISTER_LAYER_CLASS(GatedMaxAve);

} 
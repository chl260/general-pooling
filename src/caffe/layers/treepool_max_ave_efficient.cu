// Copyright 2015 Chen-Yu Lee (chl260@ucsd.edu)

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
//#include <iostream>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__device__ inline Dtype sigmoid_gpu(Dtype x) {
  return 1. / (1. + exp(-x));
}


//======================================================================
// GPU forward
//======================================================================
template <typename Dtype>
__global__ void PoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    const Dtype* weight1, Dtype* top_data) {

  //-------------------------------------------------
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);
    int w_idx = 0;
    bottom_data += (n * channels + c) * height * width;
    Dtype maxval = -FLT_MAX;
    Dtype aveval = 0;
    Dtype resp_split = Dtype(0.);  
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        resp_split += bottom_data[h * width + w] * weight1[w_idx];
        maxval = max(maxval, bottom_data[h * width + w]);
        aveval += bottom_data[h * width + w];
        w_idx++;
      }
    }
    // max pool result is saved in maxval
    // ave pool result is saved in aveval / poolsize_data[index];
    // poolsize_data[index] = (hend - hstart) * (wend - wstart);

    Dtype gating_value = sigmoid_gpu(resp_split);

    // compute gated max-ave output value and save to top_data[index]
    top_data[index] = gating_value * maxval + 
                      (Dtype(1.) - gating_value) * aveval / ( (hend - hstart) * (wend - wstart) );
  }
}


template <typename Dtype>
void GatedMaxAveLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight1 = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  // first compute splitting probabilities
  PoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      weight1, top_data);

  CUDA_POST_KERNEL_CHECK;
}


//======================================================================
// GPU backward
//======================================================================
template <typename Dtype>
__global__ void Im2col_channel(const int nthreads, const int channels, 
    const int height, const int width, const int pooled_height, const int pooled_width, 
    const int ksize, const int stride, 
    const Dtype* bottom_data, Dtype* col_data) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // note that index is for top data matrix
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int wstart = pw * stride;
    int hend = hstart + ksize;
    int wend = wstart + ksize;   
    int w_idx = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        col_data[w_idx * nthreads + index] = (h >= 0 && w >=0 && h < height && w < width) ?
          bottom_data[h * width + w] : 0;
        w_idx++;
      }
    }
  }
}


template <typename Dtype>
__global__ void PoolBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, const Dtype* weight1, 
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype* weight1_diff_data, const int count) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // note that index is for top data matrix
  if (index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride;
    int hend = min(hstart + ksize, height);
    int wstart = pw * stride;
    int wend = min(wstart + ksize, width);  
    //int bottom_offset = (n * channels + c) * height * width;
    int w_idx = 0;
    bottom_data += (n * channels + c) * height * width;

    // recompute max and ave pooling results to save memory
    Dtype maxval = -FLT_MAX;
    Dtype aveval = 0;
    Dtype resp_split = Dtype(0.);  
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        resp_split += bottom_data[h * width + w] * weight1[w_idx];
        maxval = max(maxval, bottom_data[h * width + w]);
        aveval += bottom_data[h * width + w];
        w_idx++;
      }
    }
    Dtype split_p1 = sigmoid_gpu(resp_split);
    Dtype split_p2 = Dtype(1.) - split_p1;
    Dtype top_data1 = maxval;
    int poolsize_data = (hend - hstart) * (wend - wstart);
    Dtype top_data2 = aveval / poolsize_data;

    // precompute temporary value
    Dtype temp = split_p1 * split_p2 * (top_data1 - top_data2);
    //-------------------------------------------------------
    w_idx = 0;
    bottom_diff += (n * channels + c) * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        // gradient w.r.t weight1 = g(1-g)*bottom_data*[max-ave]
        weight1_diff_data[w_idx*count + index] = temp * bottom_data[h * width + w] * top_diff[index];

        // gradient w.r.t bottom_data if needed
        //bottom_diff_data[(bottom_offset + h * width + w) * (ksize*ksize) + w_idx] = 
        //  ( temp * weight1[w_idx] + split_p1 * (bottom_data[h * width + w] == top_data1) + split_p2 / poolsize_data ) * top_diff[index];
        bottom_diff[h * width + w] += ( temp * weight1[w_idx] + split_p1 * (bottom_data[h * width + w] == top_data1) + split_p2 / poolsize_data ) * top_diff[index];

        w_idx++;
      }
    }

  } 
}



template <typename Dtype>
void GatedMaxAveLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //if (!propagate_down) {
  //  return Dtype(0.);
  //}
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //-----------------------------------------------------------  
  const Dtype* weight1 = this->blobs_[0]->gpu_data();
  Dtype* weight1_diff = this->blobs_[0]->mutable_gpu_diff();
  //Dtype* bottom_diff_data = bottom_diff_buffer_.mutable_gpu_data();
  Dtype* weight1_diff_data = weight1_diff_buffer_.mutable_gpu_data();

  int count = top[0]->count();
  int bottom_count = bottom[0]->count();

  //-----------------------------------------------------------
  //CUDA_CHECK(cudaMemset(weight1_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count()));
  //CUDA_CHECK(cudaMemset(bottom_diff_data, 0, sizeof(Dtype) * (KSIZE_*KSIZE_) * bottom[0]->count()));
  CUDA_CHECK(cudaMemset(weight1_diff_data, 0, sizeof(Dtype) * (KSIZE_*KSIZE_) * top[0]->count()));
  //-----------------------------------------------------------
 
  PoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, CHANNELS_, HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, 
      KSIZE_, STRIDE_, weight1, bottom_data,
      bottom_diff, weight1_diff_data, count);

  // accumulate correct gradients for bottom_diff
  //caffe_gpu_gemv<Dtype>(CblasNoTrans, bottom_count, (KSIZE_*KSIZE_), 1.,
  //        bottom_diff_data, reinterpret_cast<const Dtype*>(bias_multiplier2_->gpu_data()),
  //        0., bottom_diff);

  // accumulate correct gradients for weight1_diff
  caffe_gpu_gemv<Dtype>(CblasNoTrans, (KSIZE_*KSIZE_), count, 1.,
          weight1_diff_data, reinterpret_cast<const Dtype*>(bias_multiplier1_->gpu_data()),
          0., weight1_diff);

  CUDA_POST_KERNEL_CHECK;

  //return Dtype(0.);
}


//INSTANTIATE_CLASS(GatedMaxAveLayer);
INSTANTIATE_LAYER_GPU_FUNCS(GatedMaxAveLayer);

} 

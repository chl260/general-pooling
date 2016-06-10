// Copyright 2016 Chen-Yu Lee (chl260@ucsd.edu)

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
__global__ void ForestPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    const Dtype* weight1,
    Dtype* split_p1, Dtype* split_p2) {

  //-------------------------------------------------
  int index = threadIdx.x + blockIdx.x * blockDim.x;
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
    //-------------------------------------------------------
    Dtype resp_split = Dtype(0.);    
    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
          resp_split += bottom_data[h * width + w] * weight1[w_idx];
        }
        w_idx++;
      }
    }
    
    // compute splitting probability first
    split_p1[index] = sigmoid_gpu( resp_split );
    split_p2[index] = Dtype(1.) - split_p1[index];    
  }
}

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, Dtype* top_data) {
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
    Dtype maxval = -FLT_MAX;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        maxval = max(maxval, bottom_data[h * width + w]);
      }
    }
    top_data[index] = maxval;
  }  // (if index < nthreads)
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    Dtype* poolsize_data,
    Dtype* top_data) {
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
    Dtype aveval = 0;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    poolsize_data[index] = (hend - hstart) * (wend - wstart);
    top_data[index] = aveval / poolsize_data[index];
  }  // (if index < nthreads)
}

template <typename Dtype>
void TreepoolMaxAveLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight1 = this->blobs_[0]->gpu_data();
  Dtype* split_p1 = split_buffer1_.mutable_gpu_data();
  Dtype* split_p2 = split_buffer2_.mutable_gpu_data();  
  Dtype* top_data1 = top_buffer1_.mutable_gpu_data();
  Dtype* top_data2 = top_buffer2_.mutable_gpu_data();
  Dtype* mat_data = mat_buffer_.mutable_gpu_data();
  Dtype* poolsize_data = poolsize_buffer_.mutable_gpu_data();
  int count = top[0]->count();


  // first compute splitting probabilities
  ForestPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      weight1, split_p1, split_p2);

  // compute max pooling
  MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      top_data1);

  // compute ave pooling
  AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      poolsize_data,
      top_data2);

  // top_data = split_p1 * max(x) + split_p2 * ave(x)
  caffe_gpu_mul<Dtype>(count, split_p1, top_data1, top_data);
  caffe_gpu_mul<Dtype>(count, split_p2, top_data2, mat_data);
  caffe_gpu_add<Dtype>(count, top_data, mat_data, top_data);

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
    const Dtype* split_p1, const Dtype* split_p2, 
    const Dtype* top_data1, const Dtype* top_data2, 
    const Dtype* poolsize_data, const Dtype* bottom_data,
    Dtype* bottom_diff_data) {

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
    int bottom_offset = (n * channels + c) * height * width;
    int w_idx = 0;
    bottom_data += (n * channels + c) * height * width;

    // precompute temporary value
    Dtype temp = split_p1[index] * split_p2[index] * (top_data1[index] - top_data2[index]);
    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
        
          // gradient w.r.t bottom_data if needed
          bottom_diff_data[(bottom_offset + h * width + w) * (ksize*ksize) + w_idx] = 
            ( temp * weight1[w_idx] + split_p1[index] * (bottom_data[h * width + w] == top_data1[index]) + split_p2[index] / poolsize_data[index] ) * top_diff[index];

        }
        w_idx++;
      }
    }
  } 
}



template <typename Dtype>
void TreepoolMaxAveLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype* split_p1 = split_buffer1_.gpu_data();
  const Dtype* split_p2 = split_buffer2_.gpu_data();  
  const Dtype* top_data1 = top_buffer1_.gpu_data();
  const Dtype* top_data2 = top_buffer2_.gpu_data();  
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* weight1_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bottom_diff_data = bottom_diff_buffer_.mutable_gpu_data();
  Dtype* mat_data = mat_buffer_.mutable_gpu_data();
  const Dtype* poolsize_data = poolsize_buffer_.gpu_data();

  int count = top[0]->count();
  int bottom_count = bottom[0]->count();

  //-----------------------------------------------------------
  CUDA_CHECK(cudaMemset(weight1_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count()));
  CUDA_CHECK(cudaMemset(bottom_diff_data, 0, sizeof(Dtype) * (KSIZE_*KSIZE_) * bottom[0]->count()));
  //-----------------------------------------------------------
 
  PoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, CHANNELS_, HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, 
      KSIZE_, STRIDE_, weight1, split_p1, split_p2, top_data1, top_data2, 
      poolsize_data, bottom_data,
      bottom_diff_data);

  // accumulate correct gradients
  caffe_gpu_gemv<Dtype>(CblasNoTrans, bottom_count, (KSIZE_*KSIZE_), 1.,
          bottom_diff_data, reinterpret_cast<const Dtype*>(bias_multiplier2_->gpu_data()),
          0., bottom_diff);

  // ----------------------------------------------------------------
  // weight_diff
  Im2col_channel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, CHANNELS_, HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_, 
        bottom_data, col_data);

  caffe_gpu_sub<Dtype>(count, top_data1, top_data2, mat_data);
  caffe_gpu_mul<Dtype>(count, split_p1, mat_data, mat_data);
  caffe_gpu_mul<Dtype>(count, split_p2, mat_data, mat_data);
  caffe_gpu_mul<Dtype>(count, top_diff, mat_data, mat_data);
  // gradient w.r.t. weight1_diff[w_idx] = top_diff[index] * split_p1[index] * split_p2[index] * [max - ave] * bottom_data
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data, (Dtype)0., weight1_diff);


  CUDA_POST_KERNEL_CHECK;

  //return Dtype(0.);
}


//INSTANTIATE_CLASS(TreepoolMaxAveLayer);
INSTANTIATE_LAYER_GPU_FUNCS(TreepoolMaxAveLayer);

} 

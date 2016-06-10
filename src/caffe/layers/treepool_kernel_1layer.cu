// Copyright 2016 Chen-Yu Lee (chl260@ucsd.edu)

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__device__ inline Dtype sigmoid_gpu(Dtype x) {
  return 1. / (1. + exp(-x));
}

/*
template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = sigmoid_gpu(in[index]);
  }
}
*/

//======================================================================
// GPU forward
//======================================================================
template <typename Dtype>
__global__ void ForestPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    const Dtype* v1, const Dtype* v2, const Dtype* weight1,
    Dtype* split_p1, Dtype* split_p2, Dtype* resp_data1, Dtype* resp_data2,
    Dtype* top_data) {

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
    Dtype resp1 = Dtype(0.);
    Dtype resp2 = Dtype(0.);
    Dtype resp_split = Dtype(0.);    
    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
          resp1 += bottom_data[h * width + w] * v1[w_idx];
          resp2 += bottom_data[h * width + w] * v2[w_idx];
          resp_split += bottom_data[h * width + w] * weight1[w_idx];
        }
        w_idx++;
      }
    }
    
    // compute splitting probability first
    split_p1[index] = sigmoid_gpu( resp_split );
    split_p2[index] = Dtype(1.) - split_p1[index];    
    resp_data1[index] = resp1;  
    resp_data2[index] = resp2;
    top_data[index] = split_p1[index] * resp1 + split_p2[index] * resp2;
  }
}


template <typename Dtype>
void TreepoolKernel1LLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* v1 = this->blobs_[0]->gpu_data();
  const Dtype* v2 = this->blobs_[1]->gpu_data();
  const Dtype* weight1 = this->blobs_[2]->gpu_data();
  Dtype* split_p1 = split_buffer1_.mutable_gpu_data();
  Dtype* split_p2 = split_buffer2_.mutable_gpu_data();   
  Dtype* resp_data1 = resp_buffer1_.mutable_gpu_data();
  Dtype* resp_data2 = resp_buffer2_.mutable_gpu_data();  
  int count = top[0]->count();


  // NOLINT_NEXT_LINE(whitespace/operators)
  ForestPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      v1, v2, weight1, split_p1, split_p2, resp_data1, resp_data2,
      top_data);

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
__global__ void ForestPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    const Dtype* v1, const Dtype* v2, const Dtype* weight1, 
    const Dtype* split_p1, const Dtype* split_p2, 
    const Dtype* resp_data1, Dtype* resp_data2,
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
    //int bottom_w_idx_offset = num * channels * height * width;
    int w_idx = 0;

    // precompute temporary value
    Dtype temp = split_p1[index] * split_p2[index] * (resp_data1[index] - resp_data2[index]);
    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
          // gradient w.r.t bottom_data if needed
          bottom_diff_data[(bottom_offset + h * width + w) * (ksize*ksize) + w_idx] = ( temp * weight1[w_idx] + split_p1[index] * v1[w_idx] + split_p2[index] * v2[w_idx] ) * top_diff[index];
          //bottom_diff[h * width + w] += ( temp * weight1[w_idx] + split_p1[index] * v1[w_idx] + split_p2[index] * v2[w_idx] ) * top_diff[index];
        }
        w_idx++;
      }
    }
  } 
}

template <typename Dtype>
void TreepoolKernel1LLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //if (!propagate_down) {
  //  return Dtype(0.);
  //}
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //-----------------------------------------------------------  
  const Dtype* v1 = this->blobs_[0]->gpu_data();
  const Dtype* v2 = this->blobs_[1]->gpu_data();
  const Dtype* weight1 = this->blobs_[2]->gpu_data();
  const Dtype* split_p1 = split_buffer1_.gpu_data();
  const Dtype* split_p2 = split_buffer2_.gpu_data();  
  const Dtype* resp_data1 = resp_buffer1_.gpu_data();
  Dtype* resp_data2 = resp_buffer2_.mutable_gpu_data();
  Dtype* mat_data = mat_buffer_.mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* bottom_diff_data = bottom_diff_buffer_.mutable_gpu_data();
  //Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  //-----------------------------------------------------------
  Dtype* v1_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* v2_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* weight1_diff = this->blobs_[2]->mutable_gpu_diff();
  
  //int count = bottom[0]->count();
  int count = top[0]->count();

  //-----------------------------------------------------------
  CUDA_CHECK(cudaMemset(v1_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  CUDA_CHECK(cudaMemset(v2_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
  CUDA_CHECK(cudaMemset(weight1_diff, 0, sizeof(Dtype) * this->blobs_[2]->count()));
  CUDA_CHECK(cudaMemset(col_data, 0, sizeof(Dtype) * COUNT2_));
  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count()));
  CUDA_CHECK(cudaMemset(bottom_diff_data, 0, sizeof(Dtype) * (KSIZE_*KSIZE_) * bottom[0]->count()));
  //-----------------------------------------------------------
  
  ForestPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      v1, v2, weight1, split_p1, split_p2, resp_data1, resp_data2,
      bottom_diff_data);

  CUDA_POST_KERNEL_CHECK;


  
  Im2col_channel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, CHANNELS_, HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_, 
      bottom_data, col_data);
  
  CUDA_POST_KERNEL_CHECK;

  // gradient w.r.t. v1_diff[w_idx] = top_diff[index] * split_p1[index] * bottom_data[w_idx] 
  caffe_gpu_mul<Dtype>(count, split_p1, top_diff, mat_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data, (Dtype)0., v1_diff);

  // gradient w.r.t. v2_diff[w_idx] = top_diff[index] * split_p2[index] * bottom_data[w_idx] 
  caffe_gpu_mul<Dtype>(count, split_p2, top_diff, mat_data);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data, (Dtype)0., v2_diff);

  // gradient w.r.t. weight1_diff[w_idx] = top_diff[index] * split_p1[index] * split_p2[index] * 
  //                                       (resp1[index]-resp2[index]) * bottom_data[w_idx];
  caffe_gpu_mul<Dtype>(count, mat_data, split_p1, mat_data);  // mat_data = p1 * p2* top_diff
  caffe_gpu_sub<Dtype>(count, resp_data1, resp_data2, resp_data2); 
  caffe_gpu_mul<Dtype>(count, mat_data, resp_data2, mat_data);  // mat_data = p1 * p2* top_diff * (resp1-resp2)
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data, (Dtype)0., weight1_diff);

                        

  int bottom_count = bottom[0]->count();
  // accumulate correct gradients
  caffe_gpu_gemv<Dtype>(CblasNoTrans, bottom_count, (KSIZE_*KSIZE_), 1.,
          bottom_diff_data, reinterpret_cast<const Dtype*>(bias_multiplier2_->gpu_data()),
          0., bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(TreepoolKernel1LLayer);

}  // namespace caffe

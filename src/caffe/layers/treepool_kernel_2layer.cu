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


//======================================================================
// GPU forward
//======================================================================
template <typename Dtype>
__global__ void ForestPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int ksize, const int stride, 
    const Dtype* v1, const Dtype* v2, const Dtype* v3, const Dtype* v4, 
    const Dtype* weight1, const Dtype* weight2, const Dtype* weight3,
    Dtype* split_p1, Dtype* split_p2, Dtype* split_p3,
    Dtype* split_p1c, Dtype* split_p2c, Dtype* split_p3c,
    Dtype* resp_data1, Dtype* resp_data2, Dtype* resp_data3, Dtype* resp_data4,
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
    Dtype resp3 = Dtype(0.);
    Dtype resp4 = Dtype(0.);
    Dtype resp_split1 = Dtype(0.);   
    Dtype resp_split2 = Dtype(0.); 
    Dtype resp_split3 = Dtype(0.);  
    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
          resp1 += bottom_data[h * width + w] * v1[w_idx];
          resp2 += bottom_data[h * width + w] * v2[w_idx];
          resp3 += bottom_data[h * width + w] * v3[w_idx];
          resp4 += bottom_data[h * width + w] * v4[w_idx];
          resp_split1 += bottom_data[h * width + w] * weight1[w_idx];
          resp_split2 += bottom_data[h * width + w] * weight2[w_idx];
          resp_split3 += bottom_data[h * width + w] * weight3[w_idx];
        }
        w_idx++;
      }
    }
    
    // compute splitting probability first
    split_p1[index] = sigmoid_gpu( resp_split1 );
    split_p2[index] = sigmoid_gpu( resp_split2 );
    split_p3[index] = sigmoid_gpu( resp_split3 );
    split_p1c[index] = Dtype(1.) - split_p1[index];
    split_p2c[index] = Dtype(1.) - split_p2[index];
    split_p3c[index] = Dtype(1.) - split_p3[index];    
    //split_p2[index] = Dtype(1.) - split_p1[index];    
    resp_data1[index] = resp1;  
    resp_data2[index] = resp2;
    resp_data3[index] = resp3;
    resp_data4[index] = resp4;
    //top_data[index] = split_p1[index] * resp1 + split_p2[index] * resp2;
    top_data[index] = split_p3[index] * ( split_p1[index] * resp1 + split_p1c[index] * resp2 ) + 
                      split_p3c[index] * ( split_p2[index] * resp3 + split_p2c[index] * resp4 );
  }
}


template <typename Dtype>
void TreepoolKernel2LLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* v1 = this->blobs_[0]->gpu_data();
  const Dtype* v2 = this->blobs_[1]->gpu_data();
  const Dtype* v3 = this->blobs_[2]->gpu_data();
  const Dtype* v4 = this->blobs_[3]->gpu_data();
  const Dtype* weight1 = this->blobs_[4]->gpu_data();
  const Dtype* weight2 = this->blobs_[5]->gpu_data();
  const Dtype* weight3 = this->blobs_[6]->gpu_data();
  Dtype* split_p1 = split_buffer1_.mutable_gpu_data();
  Dtype* split_p2 = split_buffer2_.mutable_gpu_data();  
  Dtype* split_p3 = split_buffer3_.mutable_gpu_data(); 
  Dtype* split_p1c = split_buffer1c_.mutable_gpu_data();
  Dtype* split_p2c = split_buffer2c_.mutable_gpu_data();  
  Dtype* split_p3c = split_buffer3c_.mutable_gpu_data();   
  Dtype* resp_data1 = resp_buffer1_.mutable_gpu_data();
  Dtype* resp_data2 = resp_buffer2_.mutable_gpu_data();  
  Dtype* resp_data3 = resp_buffer3_.mutable_gpu_data();
  Dtype* resp_data4 = resp_buffer4_.mutable_gpu_data();
  int count = top[0]->count();


  // NOLINT_NEXT_LINE(whitespace/operators)
  ForestPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      v1, v2, v3, v4, weight1, weight2, weight3,
      split_p1, split_p2, split_p3, split_p1c, split_p2c, split_p3c,
      resp_data1, resp_data2, resp_data3, resp_data4,
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
    const Dtype* v1, const Dtype* v2, const Dtype* v3, const Dtype* v4, 
    const Dtype* weight1, const Dtype* weight2, const Dtype* weight3, 
    const Dtype* split_p1, const Dtype* split_p2, const Dtype* split_p3, 
    const Dtype* split_p1c, const Dtype* split_p2c, const Dtype* split_p3c, 
    const Dtype* resp_data1, const Dtype* resp_data2, const Dtype* resp_data3, const Dtype* resp_data4,
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
    // temp1 = g(w3x) * (1-g(w3x)) * [g(w1x)*resp1 - (1-g(w1x))*resp2]
    Dtype temp1 = split_p3[index] * split_p3c[index] * ( split_p1[index]*resp_data1[index] + split_p1c[index]*resp_data2[index] );
    // temp2 = g(w3x) * g(w1x) * (1-g(w1x)) * [resp1 - resp2]
    Dtype temp2 = split_p3[index] * split_p1[index] * split_p1c[index] * ( resp_data1[index] - resp_data2[index] );
    // temp4 = g(w3x) * (1-g(w3x)) * [g(w2x)*resp3 - (1-g(w2x))*resp4]
    Dtype temp4 = split_p3[index] * split_p3c[index] * ( split_p2[index]*resp_data3[index] + split_p2c[index]*resp_data4[index] );
    // temp5 = (1-g(w3x)) * g(w2x) * (1-g(w2x)) * [resp3 - resp4]
    Dtype temp5 = split_p3c[index] * split_p2[index] * split_p2c[index] * ( resp_data3[index] - resp_data4[index] );

    //-------------------------------------------------------
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if ( h < height && w < width ){
          // gradient w.r.t bottom_data if needed
          // bottom_diff = temp1*W3 + temp2*W1 + g(w3x)*(g(w1x)*v1 + (1-g(w1x)*v2)) - temp4*W3 + temp5*W2 + (1-g(w3x))*[g(w2x)*v3+(1-g(w2x)*v4)]
          // = (temp1-temp4)*W3 + temp2*W1 + temp5*W2 + g(w3x)*(g(w1x)*v1 + (1-g(w1x)*v2)) + (1-g(w3x))*[g(w2x)*v3+(1-g(w2x)*v4)]
          bottom_diff_data[(bottom_offset + h * width + w) * (ksize*ksize) + w_idx] = 
              ( (temp1-temp4)*weight3[w_idx] + temp2*weight1[w_idx] + temp5*weight2[w_idx] +
                split_p3[index] * ( split_p1[index] * v1[w_idx] + split_p1c[index] * v2[w_idx] ) + 
                split_p3c[index] * (split_p2[index] * v3[w_idx] + split_p2c[index] * v4[w_idx] ) ) * top_diff[index];
          //bottom_diff_data[(bottom_offset + h * width + w) * (ksize*ksize) + w_idx] = ( temp * weight1[w_idx] + split_p1[index] * v1[w_idx] + split_p2[index] * v2[w_idx] ) * top_diff[index];
          //bottom_diff[h * width + w] += ( temp * weight1[w_idx] + split_p1[index] * v1[w_idx] + split_p2[index] * v2[w_idx] ) * top_diff[index];
        }
        w_idx++;
      }
    }
  } 
}

template <typename Dtype>
void TreepoolKernel2LLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
  const Dtype* v3 = this->blobs_[2]->gpu_data();
  const Dtype* v4 = this->blobs_[3]->gpu_data();
  const Dtype* weight1 = this->blobs_[4]->gpu_data();
  const Dtype* weight2 = this->blobs_[5]->gpu_data();
  const Dtype* weight3 = this->blobs_[6]->gpu_data();
  const Dtype* split_p1 = split_buffer1_.gpu_data();
  const Dtype* split_p2 = split_buffer2_.gpu_data();  
  const Dtype* split_p3 = split_buffer3_.gpu_data(); 
  const Dtype* split_p1c = split_buffer1c_.gpu_data();
  const Dtype* split_p2c = split_buffer2c_.gpu_data();  
  const Dtype* split_p3c = split_buffer3c_.gpu_data();   
  const Dtype* resp_data1 = resp_buffer1_.gpu_data();
  const Dtype* resp_data2 = resp_buffer2_.gpu_data();
  const Dtype* resp_data3 = resp_buffer3_.gpu_data();
  const Dtype* resp_data4 = resp_buffer4_.gpu_data();
  //Dtype* resp_data2 = resp_buffer2_.mutable_gpu_data();
  Dtype* mat_data1 = mat_buffer1_.mutable_gpu_data();
  Dtype* mat_data2 = mat_buffer2_.mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* bottom_diff_data = bottom_diff_buffer_.mutable_gpu_data();
  //Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  //-----------------------------------------------------------
  Dtype* v1_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* v2_diff = this->blobs_[1]->mutable_gpu_diff();
  Dtype* v3_diff = this->blobs_[2]->mutable_gpu_diff();
  Dtype* v4_diff = this->blobs_[3]->mutable_gpu_diff();
  Dtype* weight1_diff = this->blobs_[4]->mutable_gpu_diff();
  Dtype* weight2_diff = this->blobs_[5]->mutable_gpu_diff();
  Dtype* weight3_diff = this->blobs_[6]->mutable_gpu_diff();
  //const Dtype* ones_data = ones_vector_.gpu_data();
  //int count = bottom[0]->count();
  int count = top[0]->count();

  //-----------------------------------------------------------
  CUDA_CHECK(cudaMemset(v1_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  CUDA_CHECK(cudaMemset(v2_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
  CUDA_CHECK(cudaMemset(v3_diff, 0, sizeof(Dtype) * this->blobs_[2]->count()));
  CUDA_CHECK(cudaMemset(v4_diff, 0, sizeof(Dtype) * this->blobs_[3]->count()));
  CUDA_CHECK(cudaMemset(weight1_diff, 0, sizeof(Dtype) * this->blobs_[4]->count()));
  CUDA_CHECK(cudaMemset(weight2_diff, 0, sizeof(Dtype) * this->blobs_[5]->count()));
  CUDA_CHECK(cudaMemset(weight3_diff, 0, sizeof(Dtype) * this->blobs_[6]->count()));
  CUDA_CHECK(cudaMemset(col_data, 0, sizeof(Dtype) * COUNT2_));
  CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count()));
  CUDA_CHECK(cudaMemset(bottom_diff_data, 0, sizeof(Dtype) * (KSIZE_*KSIZE_) * bottom[0]->count()));
  //-----------------------------------------------------------
  // gradients w.r.t. bottom_data
  ForestPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), CHANNELS_,
      HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_,
      v1, v2, v3, v4, weight1, weight2, weight3,
      split_p1, split_p2, split_p3,
      split_p1c, split_p2c, split_p3c,
      resp_data1, resp_data2, resp_data3, resp_data4,
      bottom_diff_data);

  CUDA_POST_KERNEL_CHECK;

  int bottom_count = bottom[0]->count();
  // accumulate correct gradients
  caffe_gpu_gemv<Dtype>(CblasNoTrans, bottom_count, (KSIZE_*KSIZE_), 1.,
          bottom_diff_data, reinterpret_cast<const Dtype*>(bias_multiplier2_->gpu_data()),
          0., bottom_diff);

  // -------------------------------------------------------------------------------------------
  // im2col for bottom_data
  Im2col_channel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, CHANNELS_, HEIGHT_, WIDTH_, POOLED_HEIGHT_, POOLED_WIDTH_, KSIZE_, STRIDE_, 
      bottom_data, col_data);
  
  CUDA_POST_KERNEL_CHECK;

  // gradient w.r.t. v1_diff[w_idx] = top_diff[index] * g(w3x) * g(w1x) * bottom_data[w_idx] 
  caffe_gpu_mul<Dtype>(count, split_p3, top_diff, mat_data1);
  caffe_gpu_mul<Dtype>(count, split_p1, mat_data1, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., v1_diff);

  // gradient w.r.t. v2_diff[w_idx] = top_diff[index] * g(w3x) * (1-g(w1x)) * bottom_data[w_idx] 
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p1, mat_data1); 
  caffe_gpu_mul<Dtype>(count, split_p1c, split_p3, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., v2_diff);

  // gradient w.r.t. v3_diff[w_idx] = top_diff[index] * (1-g(w3x)) * g(w2x) * bottom_data[w_idx] 
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p3, mat_data1); 
  caffe_gpu_mul<Dtype>(count, split_p3c, split_p2, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., v3_diff);

  // gradient w.r.t. v4_diff[w_idx] = top_diff[index] * (1-g(w3x)) * (1-g(w2x)) * bottom_data[w_idx] 
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p3, mat_data1); 
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p2, mat_data2); 
  caffe_gpu_mul<Dtype>(count, split_p3c, split_p2c, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., v4_diff);


  // gradient w.r.t. weight1_diff[w_idx] = top_diff[index] * g(w3x) * g(w1x) * (1-g(w1x)) * (resp1[index]-resp2[index]) * bottom_data[w_idx];
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p1, mat_data1); 
  caffe_gpu_mul<Dtype>(count, split_p1c, split_p1, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, split_p3, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  caffe_gpu_sub<Dtype>(count, resp_data1, resp_data2, mat_data2); 
  caffe_gpu_mul<Dtype>(count, mat_data1, mat_data2, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., weight1_diff);

                      
  // gradient w.r.t. weight2_diff[w_idx] = top_diff[index] * (1-g(w3x)) * g(w2x) * (1-g(w2x)) * (resp3[index]-resp4[index]) * bottom_data[w_idx];
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p2, mat_data1); 
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p3, mat_data2); 
  caffe_gpu_mul<Dtype>(count, split_p3c, split_p2c, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, split_p2, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  caffe_gpu_sub<Dtype>(count, resp_data3, resp_data4, mat_data2); 
  caffe_gpu_mul<Dtype>(count, mat_data1, mat_data2, mat_data1);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., weight2_diff);


  // gradient w.r.t. weight3_diff[w_idx] = top_diff[index] * g(w3x) * (1-g(w3x)) * 
  //                                     ( g(w1x)*resp1[index] + (1-g(w1x))*resp2 - g(w2x)*resp3 - (1-g(w2x))*resp4) * bottom_data[w_idx];
  caffe_gpu_mul<Dtype>(count, split_p1, resp_data1, mat_data1);
  //caffe_gpu_sub<Dtype>(count, ones_data, split_p1, mat_data2); 
  caffe_gpu_mul<Dtype>(count, split_p1c, resp_data2, mat_data2);
  caffe_gpu_add<Dtype>(count, mat_data1, mat_data2, mat_data1);

  caffe_gpu_mul<Dtype>(count, split_p2, resp_data3, mat_data2);
  caffe_gpu_sub<Dtype>(count, mat_data1, mat_data2, mat_data1); 

  //caffe_gpu_sub<Dtype>(count, ones_data, split_p2, mat_data2);
  caffe_gpu_mul<Dtype>(count, split_p2c, resp_data4, mat_data2);
  caffe_gpu_sub<Dtype>(count, mat_data1, mat_data2, mat_data1);

  caffe_gpu_mul<Dtype>(count, mat_data1, split_p3, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, split_p3c, mat_data1);
  caffe_gpu_mul<Dtype>(count, mat_data1, top_diff, mat_data1);
  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, (KSIZE_*KSIZE_), 1, count, (Dtype)1.,
                        col_data, mat_data1, (Dtype)0., weight3_diff);


  //return Dtype(0.);
}


INSTANTIATE_LAYER_GPU_FUNCS(TreepoolKernel2LLayer);


}  // namespace caffe

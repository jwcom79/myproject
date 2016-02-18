#include <vector>

#include "caffe/layers/pushin_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

		template <typename Dtype>
		__global__ void PushinForward(const int n, const Dtype* in,
																	unsigned int* mask, const unsigned int threshold, const float scale,
																	Dtype* out, int st_count) {
						for (int i = 0; i < st_count; ++i)
								mask[i] = 1;
						for (int i = st_count; i < n; ++i)
								mask[i] = 0;

				CUDA_KERNEL_LOOP(index, n) {
						out[index] = in[index] * (mask[index] > threshold) * scale;
				}
		}

		template <typename Dtype>
		void PushinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
														 						 const vector<Blob<Dtype>*>& top) {
				const Dtype* bottom_data = bottom[0]->gpu_data();
				Dtype* top_data = top[0]->mutable_gpu_data();
				const int count = bottom[0]->count();
				//if (this->phase_ == TRAIN) {
						unsigned int* mask =
								static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
						//caffe_gpu_rng_uniform(count, mask);
						// set thresholds
						// NOLINT_NEXT_LINE(whitespace/operators)
						
						
						PushinForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
										count, bottom_data, mask, uint_thres_, scale_, top_data, st_count);
						CUDA_POST_KERNEL_CHECK;
				//} else {
				//		caffe_copy(count, bottom_data, top_data);
				//}
						st_count += 20;

						if(st_count > count)
								st_count = count;
		}

		template <typename Dtype>
		__global__ void PushinBackward(const int n, const Dtype* in_diff,
																	 unsigned int* mask, const unsigned int threshold, const float scale,
																	 Dtype* out_diff, int st_count) {
								for (int i = 0; i < st_count; ++i)
										mask[i] = 1;
								for (int i = st_count; i < n; ++i)
										mask[i] = 0;

				CUDA_KERNEL_LOOP(index, n) {
						out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
				}
		}

		template <typename Dtype>
		void PushinLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
																					const vector<bool>& propagate_down,
																					const vector<Blob<Dtype>*>& bottom) {
				if (propagate_down[0]) {
						const Dtype* top_diff = top[0]->gpu_diff();
						Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
						//if (this->phase_ == TRAIN) {
								unsigned int* mask =
										static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
								//const unsigned int* mask =
								//		static_cast<const unsigned int*>(rand_vec_.gpu_data());
								const int count = bottom[0]->count();
								// NOLINT_NEXT_LINE(whitespace/operators)
								//
								//
								PushinBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
										CAFFE_CUDA_NUM_THREADS>>>(
														//count, top_diff, mask, uint_thres_, scale_, bottom_diff);
														count, top_diff, mask, uint_thres_, scale_, bottom_diff, st_count);
								CUDA_POST_KERNEL_CHECK;
						//} else {
						//		caffe_copy(top[0]->count(), top_diff, bottom_diff);
						//}
								st_count += 20;

								if(st_count > count)
										st_count = count;
				}
		}

		INSTANTIATE_LAYER_GPU_FUNCS(PushinLayer);

}  // namespace caffe

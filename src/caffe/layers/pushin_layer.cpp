#include <vector>

#include "caffe/layers/pushin_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

		template <typename Dtype>
		void PushinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
																				const vector<Blob<Dtype>*>& top){
				NeuronLayer<Dtype>::LayerSetUp(bottom, top);
				threshold_ = this->layer_param_.pushin_param().pushin_ratio();
				DCHECK(threshold_ > 0.);
				DCHECK(threshold_ < 1.);
				scale_ = 1. / (1. - threshold_);
				uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
				st_count = bottom[0]->count() * threshold_;
		}

		template <typename Dtype>
		void PushinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
																		 const vector<Blob<Dtype>*>& top){
				NeuronLayer<Dtype>::Reshape(bottom, top);
				// Set up the cache for random number generation
				rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
								bottom[0]->height(), bottom[0]->width());
		}

		template <typename Dtype>
		void PushinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
																				 const vector<Blob<Dtype>*>& top){
				const Dtype* bottom_data = bottom[0]->cpu_data();
				Dtype* top_data = top[0]->mutable_cpu_data();
				unsigned int* mask = rand_vec_.mutable_cpu_data();
				const int count = bottom[0]->count();

				//if (this->phase_ == TRAIN) {
				// Create random numbers
				/*
					 caffe_rng_bernoulli(count, 1. - threshold_, mask);
					 for (int i = 0; i < count; ++i) {
					 top_data[i] = bottom_data[i] * mask[i] * scale_;
					 }
					 */
				for (int i = 0; i < st_count; ++i) {
						mask[i] = 1;
				}
				for (int i = st_count; i < count; ++i) {
						mask[i] = 0;
				}
				for (int i = 0; i < count; ++i) {
						top_data[i] = bottom_data[i] * mask[i] * scale_;
				}
				//} else {
				//caffe_copy(bottom[0]->count(), bottom_data, top_data);
				//		for (int i = 0; i < count; ++i) {
				//				top_data[i] = bottom_data[i] * mask[i] * scale_;
				//		}
				//}
				st_count++;
				if (st_count > count)
						st_count -= 1;
		}


		template <typename Dtype>
		void PushinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
																					const vector<bool>& propagate_down,
																					const vector<Blob<Dtype>*>& bottom){
				//if (propagate_down[0]) {
				const Dtype* top_diff = top[0]->cpu_diff();
				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				unsigned int* mask = rand_vec_.mutable_cpu_data();
				const int count = bottom[0]->count();
				//if (this->phase_ == TRAIN) {
				//		const unsigned int* mask = rand_vec_.cpu_data();
				for (int i = 0; i < st_count; ++i) {
						mask[i] = 1;
				}
				for (int i = st_count; i < count; ++i) {
						mask[i] = 0;
				}
				for (int i = 0; i < count; ++i) {
						bottom_diff[i] = top_diff[i] * mask[i] * scale_;
				}
				//} else {
				//		caffe_copy(top[0]->count(), top_diff, bottom_diff);
				//}
				//}
		}


#ifdef CPU_ONLY
		STUB_GPU(PushinLayer);
#endif

		INSTANTIATE_CLASS(PushinLayer);
		REGISTER_LAYER_CLASS(Pushin);

		// namespace caffe

}

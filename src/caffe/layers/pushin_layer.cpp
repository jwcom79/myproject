#include <vector>

#include "caffe/layers/pushin_layer.hpp"
#include "caffe/util/math_functions.hpp"
using namespace std;
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
				first_test = 0;
			
				//check INITIAL st_count
				cout << "INITIAL threshold_ : " << threshold_ << endl;
				cout << "INITIAL st_count: " << st_count << endl;
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

				/*
				caffe_rng_bernoulli(count, 1. - threshold_, mask);
				for (int i = 0; i < count; ++i) {
						top_data[i] = bottom_data[i] * mask[i] * scale_;
				}
				*/

				if (this->phase_ == TRAIN) {
						// Create random numbers
						for (int i = 0; i < st_count; ++i) {
								mask[i] = 1;
						}
						for (int i = st_count; i < count; ++i) {
								mask[i] = 0;
						}
						for (int i = 0; i < count; ++i) {
								top_data[i] = bottom_data[i] * mask[i] * scale_;
						}
						
						st_count += 8;

						if (st_count >= count)
								st_count = count;
						else
								cout << "on train process :" << st_count << endl;
				}

				else {
						if(first_test < 100)
								first_test += 1;

						else{
								for (int i = 0; i < st_count; ++i) {
										mask[i] = 1;
								}
								for (int i = st_count; i < count; ++i) {
										mask[i] = 0;
								}
								for (int i = 0; i < count; ++i) {
										top_data[i] = bottom_data[i] * mask[i] * scale_;
								}

								st_count += 40;

								if (st_count >= count)
										st_count = count;
								else
										cout << "on test process :" << st_count << endl;

								int num = 0;

								for (int i = 0; i < count; ++i){					
										if(mask[i])
												num += 1;
								}

								cout << "1 in mask set : " << num << endl;

								//caffe_copy(bottom[0]->count(), bottom_data, top_data);
								//for (int i = 0; i < count; ++i) 
								//		top_data[i] = bottom_data[i] * mask[i] * scale_;
						}
				}
		
						
		}


		template <typename Dtype>
		void PushinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
																					const vector<bool>& propagate_down,
																					const vector<Blob<Dtype>*>& bottom){
				if (propagate_down[0]) {
						const Dtype* top_diff = top[0]->cpu_diff();
						Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
						unsigned int* mask = rand_vec_.mutable_cpu_data();
						const int count = bottom[0]->count();
						if (this->phase_ == TRAIN) {
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

								cout << "on train Backward process :" << st_count << endl;
						}
						//
						//st_count += 20;

						//if (st_count > count)
						//		st_count = count;
						else {
								caffe_copy(top[0]->count(), top_diff, bottom_diff);
								
								cout << "on test Backward process :" << st_count << endl;
						}
				}
		}


#ifdef CPU_ONLY
		STUB_GPU(PushinLayer);
#endif

		INSTANTIATE_CLASS(PushinLayer);
		REGISTER_LAYER_CLASS(Pushin);

		// namespace caffe

}

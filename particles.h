#ifndef __PARTICLES_H__
#define __PARTICLES_H__

#include <opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// ÿ�����ӽṹ��
struct PARTICLE {
	double x;			// ��ǰx���꣨���ģ�
	double y;			// ��ǰy���꣨���ģ�
	double scale;		// ���ڱ���ϵ��
	double xPre;		// x����Ԥ��λ��
	double yPre;		// y����Ԥ��λ��
	double scalePre;	// ����Ԥ�����ϵ��
	double xOri;		// ԭʼx����
	double yOri;		// ԭʼy����
	int width;			// ԭʼ������
	int height;			// ԭʼ����߶�
	double weight;		// �����ӵ�Ȩ��
};

void particle_init(vector<PARTICLE> &particles, MatND &hist, const Rect &init_bbox);
void transition(PARTICLE &p, int w, int h, cv::RNG &rng, const double A1, const double A2, const double B0, const double TRANS_X_STD, const double TRANS_Y_STD, const double TRANS_S_STD);
MatND get_norm_HShist(Mat &rgb_img, Mat &hsv_img, const vector<int>& hist_channels, const vector<int>& hist_size, const vector<float>& hist_ranges);
void normalize_weights(vector<PARTICLE> &particles);

void resample(vector<PARTICLE> &particles, vector<PARTICLE> &new_particles, cv::RNG &rng);
int get_min_index(const vector<double>& vec, double val);
void des_sort(vector<PARTICLE> &particles);
Rect get_ROI_rect_from_frame(double x, double y, int width, int height, int frame_cols, int frame_rows, double scale);
Rect get_predict_bbox(vector<PARTICLE> &particles, const int frame_rows, const int frame_cols, bool weighted, double use_ratio);
#endif

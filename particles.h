#ifndef __PARTICLES_H__
#define __PARTICLES_H__

#include <opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// 每个粒子结构体
struct PARTICLE {
	double x;			// 当前x坐标（中心）
	double y;			// 当前y坐标（中心）
	double scale;		// 窗口比例系数
	double xPre;		// x坐标预测位置
	double yPre;		// y坐标预测位置
	double scalePre;	// 窗口预测比例系数
	double xOri;		// 原始x坐标
	double yOri;		// 原始y坐标
	int width;			// 原始区域宽度
	int height;			// 原始区域高度
	double weight;		// 该粒子的权重
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

#ifndef __FRAME_DIFFERENCE_H__
#define __FRAME_DIFFERENCE_H__

#include <opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class FrameDiff{

private:
	double th;
	int num_frame;
	double ROI_search_ratio;	// ROI���������ʼ������Χ��<0Ϊȫͼ����
	Mat current_frame;
	Mat pre_frame, pre_pre_frame;
	Mat diff_pre, diff_current;
	Mat search_mask;
	Mat morphology_kernel;
	Mat obj_mask;
	void get_search_mask(const int cols, const int rows, const Rect& obj_bbox);

public:
	FrameDiff(double th, double ROI_search_ratio);
	Mat get_object_mask(Mat &frame, const Rect& pre_obj_bbox);
};

#endif
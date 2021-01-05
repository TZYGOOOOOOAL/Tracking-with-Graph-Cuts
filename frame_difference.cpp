#include "frame_difference.h"

// *** �ο� https://www.jb51.net/article/183205.htm

// ROI_expand_ratio < 0.0 ��ʾȫͼ֡������
FrameDiff::FrameDiff(double th, double ROI_search_ratio)
{
	assert(ROI_search_ratio > 1.0 || ROI_search_ratio < 0.0);
	this->th = th;
	this->ROI_search_ratio = ROI_search_ratio;
	this->num_frame = 0;
	this->morphology_kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
}


// �õ���Ҫ����֡���ROI, ��һ����Χ��ROI�� ��ֹȫͼ����
void FrameDiff::get_search_mask(const int cols, const int rows, const Rect& obj_bbox)
{
	// ȫͼ����
	if (ROI_search_ratio < 0.0)
	{
		search_mask.setTo(255);
		return;
	}

	// ȷ��ROI_rectλ��
	Rect ROI_rect;
	double obj_center_x = obj_bbox.x + obj_bbox.width / 2.0;
	double obj_center_y = obj_bbox.y + obj_bbox.height / 2.0;

	int x1, y1, x2, y2;
	double w = obj_bbox.width * ROI_search_ratio;
	double h = obj_bbox.height * ROI_search_ratio;
	double len = std::max(w, h);

	x1 = std::max(0, cvRound(obj_center_x - len / 2.0));
	x2 = std::min(cols - 1, cvRound(obj_center_x + len / 2.0));
	y1 = std::max(0, cvRound(obj_center_y - len / 2.0));
	y2 = std::min(rows - 1, cvRound(obj_center_y + len / 2.0));

	ROI_rect.x = x1;
	ROI_rect.y = y1;
	ROI_rect.width = x2 - x1 + 1;
	ROI_rect.height = y2 - y1 + 1;

	// ����
	search_mask.setTo(0);

	// ROI��1
	search_mask(ROI_rect).setTo(255);
	return;
}


// ��֡��õ��˶�Ŀ��
Mat FrameDiff::get_object_mask(Mat &frame, const Rect& pre_obj_bbox)
{
	vector<Rect> bboxes;
	
	// ��ʼ֡
	if (num_frame == 0)
	{
		// ��ʼ��mask
		search_mask = Mat::zeros(frame.size(), CV_8UC1);

		// ת�Ҷȴ洢
		cvtColor(frame, pre_pre_frame, COLOR_BGR2GRAY);
		num_frame++;
	}
	else if (num_frame == 1)
	{
		// ���浱ǰ֡�ĻҶ�ͼ  
		cvtColor(frame, pre_frame, COLOR_BGR2GRAY);

		// ֡�Χ����
		get_search_mask(frame.cols, frame.rows, pre_obj_bbox);

		// pre - pre_pre   
		subtract(pre_frame, pre_pre_frame, diff_pre, search_mask, CV_16SC1);

		// ȡ����ֵ   
		diff_pre = cv::abs(diff_pre);
		// λ��ĸı�   
		diff_pre.convertTo(diff_pre, CV_8UC1, 1, 0);

		// ��ֵ����  
		threshold(diff_pre, diff_pre, th, 255, cv::THRESH_BINARY);
		
		num_frame++;
	}

	/*******  ��ʽ��֡��  *******/
	else
	{
		cvtColor(frame, current_frame, COLOR_BGR2GRAY);

		// ֡�Χ����
		get_search_mask(frame.cols, frame.rows, pre_obj_bbox);

		// current - pre   
		subtract(current_frame, pre_frame, diff_current, search_mask, CV_16SC1);

		//ȡ����ֵ  
		diff_current = abs(diff_current);
		diff_current.convertTo(diff_current, CV_8UC1, 1, 0);

		//��ֵ����   
		threshold(diff_current, diff_current, th, 255, THRESH_BINARY);

		//������   
		bitwise_and(diff_pre, diff_current, obj_mask);

		//��ֵ�˲� & ��̬ѧ����(������)
		//medianBlur(obj_mask, obj_mask, 3);
		morphologyEx(obj_mask, obj_mask, MORPH_CLOSE, morphology_kernel, Point(-1, -1), 2, BORDER_REPLICATE);

		// ��������ֻ�������࣬ȫ����洢��   
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(obj_mask.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		// �洢�˶���������
		obj_mask.setTo(0);
		drawContours(obj_mask, contours, -1, Scalar(255), -1);		// -1 ȫ������

		// �����滻
		pre_pre_frame = pre_frame.clone();
		pre_frame = current_frame.clone();

		diff_pre = diff_current.clone();
	}

	return obj_mask;
}


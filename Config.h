# ifndef __CONFIG_H__
# define __CONFIG_H__

#include <opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct Config
{

	int GRAPH_CUT_INTERVAL = 100000;
	int FRAME_DIFF_INTERVAL = 2;

	/****** 数据 ******/
	string DATA_DIR = "data/";
	string IMG_FMT = ".jpg";
	string VIDEO_FMT = ".avi";

	double VIDEO_FPS = 20;


	/****** 粒子群算法参数 ******/
	int NUM_PARTICLE = 50;
	/* standard deviations for gaussian sampling in transition model */
	double TRANS_X_STD = 0.5;   // 1.0
	double TRANS_Y_STD = 0.2;	// 0.5
	double TRANS_S_STD = 0.01; // 0.001
	/* autoregressive dynamics parameters for transition model */
	double A1 = 2.0;          // 2.0
	double A2 = -1.0;		  // -1.0
	double B0 = 3.0;		  // 1.0  2.0
	
	bool PREDICT_WITH_WEIGHT = true;
	double PREDICT_USE_RATIO = 0.3;    // 0.4


	/****** Graph Cut 参数 ******/
	int GRAPH_CUT_ITERS_COUNT = 1;


	/****** HS直方图参数 ******/
	// 直方图bin数量
	int HIST_H_BINS = 10, HIST_S_BINS = 10;		//180 256 10,   10 10
	vector<int> HIST_SIZE = vector<int>({ HIST_H_BINS, HIST_S_BINS });
	// 亮度h的范围
	vector<float> HIST_H_RANGE = vector<float>({ 0, 180 });
	// 饱和度s的范围
	vector<float> HIST_S_RANGE = vector<float>({ 0, 256 });
	//只比较hsv的h和s两个通道
	vector<float> HIST_RANGES = vector <float>( { HIST_H_RANGE[0], HIST_H_RANGE[1], HIST_S_RANGE[0], HIST_S_RANGE[1]});
	vector<int> HIST_CHANNELS = vector<int>({ 0, 1 });

	/****** 可视化 ******/
	bool VISUAL_TEMP_RESULT = false;
	Scalar VISUAL_PREDICT_BBOX_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_PREDICT_MASK_COLOR = Scalar(0, 0, 255);
	Scalar VISUAL_TARGET_BBOX_COLOR = Scalar(255, 0, 0);
	double VISUAL_MASK_WEIGHT = 0.7;

	/****** 帧差法参数 ******/
	double FRAME_DIFF_TH = 20;
	double FRAME_DIFF_SEARCH_RATIO = 2.0;

	/****** 两算法权重 ******/
	double LAMBDA_FRAME_DIFF = 0.5;

	/****** 评价 ******/
	double EVAL_PIXEL_DIFF_TH = 20;
	double EVAL_IOU_TH = 0.5;

	/****** 预测 ******/
	bool TEST_WITH_LABEL = true;
	bool SAVE_RESULT = false;
	string SAVE_DIR_PATH = "result";
};

# endif
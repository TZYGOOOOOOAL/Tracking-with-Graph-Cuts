#ifndef __MY_TRACK_H__
#define __MY_TRACK_H__

#include <opencv.hpp>
#include <vector>
#include <string>
#include <ctime>
#include "Config.h"
#include "particles.h"

using namespace std;
using namespace cv;

vector<Rect> track(string video_path, Rect init_bbox, const Config& config, 
	const vector<Rect>& annotation_bboxes = vector<Rect>());

void merge_particles_frameDiff_weights(vector<PARTICLE>& particles, Mat& obj_mask, double frame_diff_lambda);
#endif

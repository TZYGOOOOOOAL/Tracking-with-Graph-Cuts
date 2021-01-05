#ifndef __MAKE_DATA_H__
#define __MAKE_DATA_H__

#include <opencv.hpp>
#include <vector>
#include <string>
#include "Config.h"

using namespace std;
using namespace cv;

void imgs2video(const vector<string> &img_paths, string video_save_path, double fps);
void make_video(const Config &config);
void show_annotations(const Config &config);
#endif
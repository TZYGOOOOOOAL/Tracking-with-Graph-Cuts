#ifndef __EVAL_H__
#define __EVAL_H__

#include <opencv.hpp>
#include <vector>
#include <algorithm>
#include "Config.h"

using namespace std;
using namespace cv;

void eval(const vector<Rect>& predicts, const vector<Rect>& targets, const Config& config);

void eval_by_txts(Config &config);
#endif

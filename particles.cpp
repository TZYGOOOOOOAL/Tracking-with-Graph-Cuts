#include "particles.h"
#include <ctime>

// *** �ο� https://blog.csdn.net/dieju8330/article/details/85061448

void particle_init(vector<PARTICLE> &particles, MatND &hist, const Rect &init_bbox)
{
	//�������ӳ�ʼ�������е�Ŀ������
	for (int i = 0; i < particles.size(); i++)
	{
		particles[i].x = init_bbox.x + 0.5 * init_bbox.width;
		particles[i].y = init_bbox.y + 0.5 * init_bbox.height;
		particles[i].xPre = particles[i].x;
		particles[i].yPre = particles[i].y;
		particles[i].xOri = particles[i].x;
		particles[i].yOri = particles[i].y;
		particles[i].width = init_bbox.width;
		particles[i].height = init_bbox.height;
		particles[i].scale = 1.0;
		particles[i].scalePre = 1.0;
		particles[i].weight = 0;
	}
}

// ״̬ת��
void transition(PARTICLE &p, int w, int h, cv::RNG &rng, const double A1, const double A2,
	const double B0, const double TRANS_X_STD, const double TRANS_Y_STD, const double TRANS_S_STD)
{
	double x, y, s;
	PARTICLE pn;

	/* sample new state using second-order autoregressive dynamics */
	x = A1 * (p.x - p.xOri) + A2 * (p.xPre - p.xOri) +
		B0 *rng.gaussian(TRANS_X_STD) + p.xOri;		// �����������һʱ�̵�x
	pn.x = MAX(0.0, MIN(w - 1.0, x));				// ������ͼ��Χ
	

	y = A1 * (p.y - p.yOri) + A2 * (p.yPre - p.yOri) +
		B0 * rng.gaussian(TRANS_Y_STD) + p.yOri;
	pn.y = MAX(0.0, MIN(h - 1.0, y));

	s = A1 * (p.scale - 1.0) + A2 * (p.scalePre - 1.0) +
		B0 *  rng.gaussian(TRANS_S_STD) + 1.0;
	pn.scale = MAX(0.95*p.scale, MIN(s, 1.05*p.scale));
	//pn.scale = MAX(0.5, s);

	pn.xPre = p.x;
	pn.yPre = p.y;
	pn.scalePre = p.scale;
	pn.xOri = p.xOri;
	pn.yOri = p.yOri;
	pn.width = p.width;
	pn.height = p.height;
	pn.weight = 0;

	p = pn;
	return;
}

// �õ���һ����HSֱ��ͼ
MatND get_norm_HShist(Mat &rgb_img, Mat &hsv_img, const vector<int>& hist_channels, const vector<int>& hist_size,
	const vector<float>& hist_ranges)
{
	MatND hist;

	//Ŀ������ת����hsv�ռ�
	cvtColor(rgb_img, hsv_img, COLOR_BGR2HSV);

	//����Ŀ�������ֱ��ͼ �� ��һ��
	calcHist(vector<Mat>({ hsv_img }), hist_channels, Mat(), hist, hist_size, hist_ranges);
	normalize(hist, hist, 0, 1, NORM_MINMAX);

	return hist;
}

// ��һ��Ȩ��
void normalize_weights(vector<PARTICLE> &particles)
{
	double sum = 0;
	int n = particles.size();
	int i;

	for (i = 0; i < n; i++)
		sum += particles[i].weight;
	for (i = 0; i < n; i++)
		particles[i].weight /= sum;

	return;
}

// ****** �ز��� ******
void resample(vector<PARTICLE> &particles, vector<PARTICLE> &new_particles, cv::RNG &rng)
{
	// ����ÿ�����ӵĸ����ۼƺ�
	int num_particles = particles.size();
	vector<double> accumulate(num_particles, 0);
	double temp_sum = 0;
	int k = 0;

	// ע�������ǰ�weights����
	for (int j = num_particles - 1; j >= 0; j--){
		temp_sum += particles[j].weight;
		accumulate[j] = temp_sum;
	}

	//�����Ӹ����ۻ����������ҵ���С�Ĵ��ڸ�������������������Ƹ�����������һ�ε��µ����������� ����Ȩ�ظߵ����ӿ�ʼ��
	for (int j = 0; j < num_particles; j++){
		int copy_index = get_min_index(accumulate, rng.uniform(0.0, 1.0));
		new_particles[k++] = particles[copy_index];
		if (k == num_particles)
			break;
	}

	//�������Ĳ�����ɣ����������������������ԭ������������������Ȩ����ߵ����ӣ�ֱ�����������
	while (k < num_particles)
	{
		new_particles[k++] = particles[0]; //����Ȩֵ��ߵ�����
	}

	//�����������鸲�Ǿɵ���������
	particles = new_particles;
}


/* ���ַ��������д��ڸ���ֵ����Сֵ���� �����齵�� */
int get_min_index(const vector<double>& vec, double val)
{
	int size = vec.size();
	int right = size - 1;
	int left = 0;

	int m_idx = (left + right)/2;
	int r_idx = right;

	// ���жϼ�ֵ
	if (vec[0] <= val){
		return 0;
	}
	if (vec[size - 1] > val){
		return size - 1;
	}
	while( m_idx != r_idx ){
		r_idx = m_idx;
		if (vec[m_idx] > val){
			m_idx = (right + m_idx) / 2;
			left = r_idx;
		}
		else if (vec[m_idx] < val){
			m_idx = (left + m_idx) / 2;
			right = r_idx;
		}
		else if (vec[m_idx] == val){
			m_idx--;
			break;
		}
	}
	return m_idx;
}


// ��������
bool camp_fun(const PARTICLE& p1, const PARTICLE& p2)
{
	return p1.weight > p2.weight; 
}

void des_sort(vector<PARTICLE> &particles)
{
	//auto camp_fun = [](const PARTICLE& p1, const PARTICLE& p2)->bool{p1.weight > p2.weight; };
	std::sort(particles.begin(), particles.end(), camp_fun);
	return;
}


// ԭͼ���Ͻ�ȡROI
Rect get_ROI_rect_from_frame(double x, double y, int width, int height, int frame_cols, int frame_rows, double scale)
{
	int rect_x = std::max(0, std::min(cvRound(x - 0.5 * width * scale), cvRound(frame_cols - width * scale)));
	int rect_y = std::max(0, std::min(cvRound(y - 0.5 * height * scale), cvRound(frame_rows - height * scale)));
	int rect_width = std::min(cvRound(width * scale), frame_cols);
	int rect_height = std::min(cvRound(height * scale), frame_rows);
	return Rect(rect_x, rect_y, rect_width, rect_height);
}


// �õ�Ԥ���
Rect get_predict_bbox(vector<PARTICLE> &particles,const int frame_rows, const int frame_cols, 
	bool weighted, double use_ratio)
{
	double s;
	double x, y;
	int width = particles[0].width;
	int height = particles[0].height;

	// ���ü�Ȩ��ֱ�������Ȩ��
	if (!weighted)
	{
		x = particles[0].x;
		y = particles[0].y;
		s = particles[0].scale;
	}

	// ǰ1/3��������Ȩ
	else
	{
		x = y = s = 0.0;
		double w;
		int total_size = particles.size();
		int size = int(use_ratio * total_size + 0.5);
		double accumulate_weights = 0.0;

		// �ۻ� + Ȩ���ۻ�
		for (int i = 0; i < size; i++)
		{
			w = particles[i].weight;

			x += particles[i].x * w;
			y += particles[i].y * w;
			s += particles[i].scale * w;

			accumulate_weights += w;
		}

		// ��Ȩ��ֵ
		x /= accumulate_weights;
		y /= accumulate_weights;
		s /= accumulate_weights;
	}

	return get_ROI_rect_from_frame(x, y, width, height, frame_cols, frame_rows, s);
}

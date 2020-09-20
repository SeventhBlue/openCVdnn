#pragma once
#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class OpenPose {
public:
	OpenPose(std::string& modelPath, std::string& configPath, std::string& dataType);
	~OpenPose();
	int loadModel();
	int runningOpenPose(cv::Mat& img, std::vector<cv::Point>& openPoseRet);
	void showDetectRet(cv::Mat& img, std::vector<cv::Point>& openPoseRet);
	void saveVider(cv::Mat img, std::vector<cv::Point>& openPoseRet);

private:
	const int POSE_PAIRS[3][20][2] = {
		{   // COCO body
			{1,2}, {1,5}, {2,3},
			{3,4}, {5,6}, {6,7},
			{1,8}, {8,9}, {9,10},
			{1,11}, {11,12}, {12,13},
			{1,0}, {0,14},
			{14,16}, {0,15}, {15,17}
		},
		{   // MPI body
			{0,1}, {1,2}, {2,3},
			{3,4}, {1,5}, {5,6},
			{6,7}, {1,14}, {14,8}, {8,9},
			{9,10}, {14,11}, {11,12}, {12,13}
		},
		{   // hand
			{0,1}, {1,2}, {2,3}, {3,4},         // thumb
			{0,5}, {5,6}, {6,7}, {7,8},         // pinkie
			{0,9}, {9,10}, {10,11}, {11,12},    // middle
			{0,13}, {13,14}, {14,15}, {15,16},  // ring
			{0,17}, {17,18}, {18,19}, {19,20}   // small
		} 
	};

	std::string m_modelPath;
	std::string m_configPath;
	int m_midx, m_npairs, m_nparts;
	int m_w = 368;
	int m_h = 368;
	int m_resultW;
	int m_resultH;
	float m_thresh = 0.1;
	float m_scale = 0.003922;
	cv::dnn::Net m_net;


	// 检测的图片保存成视频的参数
	int m_saveH = 0;
	int m_saveW = 0;
	cv::VideoWriter m_viderWriter;
	std::string m_viderName;
	int m_frames = 0;

	void postprocess(cv::Mat& img, cv::Mat& result, std::vector<cv::Point>& openPoseRet);
	std::string getLocNameTime();                    // 返回格式化时间：20200426_150925
	void setViderWriterPara(const cv::Mat& img);
};

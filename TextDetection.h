#pragma once
#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef struct TextDetSt {
	std::string text;
	cv::Point2f vertices[4];   // vertices[0]:���½�   vertices[1]:���Ͻ�   vertices[2]:���Ͻ�   vertices[3]:���½�
}TextDetSt;

class TextDetection {
public:
	TextDetection(cv::String& modelDecoder, cv::String& modelRecognition, cv::String& wordDict);
	~TextDetection();

	int loadModel();
	int runningTextDetection(cv::Mat& img, std::vector<TextDetSt>& textDetRet);
	void showDetectRet(cv::Mat& img, std::vector<TextDetSt>& textDetRet);
	void saveVider(cv::Mat img, std::vector<TextDetSt>& textDetRet);

private:
	float m_confThreshold = 0.5;
	float m_nmsThreshold = 0.4;
	int m_inpWidth = 320;
	int m_inpHeight = 320;
	cv::String m_EASTModelPath;
	cv::String m_CRNNModelPath;
	cv::String m_wordDict;
	std::vector<cv::Mat> m_outs;
	cv::Mat m_blob;
	std::vector<cv::String> m_outNames;

	cv::dnn::Net m_east;
	cv::dnn::Net m_crnn;

	// ����ͼƬ�������Ƶ�Ĳ���
	int m_saveH = 0;
	int m_saveW = 0;
	cv::VideoWriter m_viderWriter;
	std::string m_viderName;
	int m_frames = 0;

	std::string getLocNameTime();                    // ���ظ�ʽ��ʱ�䣺20200426_150925
	void setViderWriterPara(const cv::Mat& img);


	void decodeBoundingBoxes(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
		std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences);
	void fourPointsTransform(const cv::Mat& frame, cv::Point2f vertices[4], cv::Mat& result);
	void decodeText(const cv::Mat& scores, std::string& text);
};
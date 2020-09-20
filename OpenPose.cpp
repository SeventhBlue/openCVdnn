#include "OpenPose.h"

OpenPose::OpenPose(std::string& modelPath, std::string& configPath, std::string& dataType) {
	m_modelPath = modelPath;
	m_configPath = configPath;
	if (dataType == "MPI") {
		m_midx = 1;
		m_npairs = 14;
		m_nparts = 16;
	}
	else if (dataType == "HAND") {
		m_midx = 2;
		m_npairs = 20;
		m_nparts = 22;
	}
	else {
		m_midx = 0;
		m_npairs = 17;
		m_nparts = 18;
	}
}

OpenPose::~OpenPose() {
	m_viderWriter.release();
}

int OpenPose::loadModel() {
	// read the network model
	m_net = cv::dnn::readNet(m_modelPath, m_configPath);
	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	return 0;
}

int OpenPose::runningOpenPose(cv::Mat& img, std::vector<cv::Point>& openPoseRet) {
	// send it through the network
	cv::Mat inputBlob = cv::dnn::blobFromImage(img, m_scale, cv::Size(m_w, m_h), cv::Scalar(0, 0, 0), false, false);
	m_net.setInput(inputBlob);
	cv::Mat result = m_net.forward();
	postprocess(img, result, openPoseRet);
}

void OpenPose::postprocess(cv::Mat& img, cv::Mat& result, std::vector<cv::Point>& openPoseRet) {
	m_resultH = result.size[2];
	m_resultW = result.size[3];
	// find the position of the body parts
	for (int n = 0; n < m_nparts; n++)
	{
		// Slice heatmap of corresponding body's part.
		cv::Mat heatMap(m_resultH, m_resultW, CV_32F, result.ptr(0, n));
		// 1 maximum per heatmap
		cv::Point p(-1, -1), pm;
		double conf;
		minMaxLoc(heatMap, 0, &conf, 0, &pm);
		if (conf > m_thresh)
			p = pm;
		openPoseRet[n] = p;
	}
}

void OpenPose::showDetectRet(cv::Mat& img, std::vector<cv::Point>& openPoseRet) {
	// connect body parts and draw it !
	float SX = float(img.cols) / m_resultW;
	float SY = float(img.rows) / m_resultH;
	for (int n = 0; n < m_npairs; n++)
	{
		// lookup 2 connected body/hand parts
		cv::Point2f a = openPoseRet[POSE_PAIRS[m_midx][n][0]];
		cv::Point2f b = openPoseRet[POSE_PAIRS[m_midx][n][1]];
		// we did not find enough confidence before
		if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
			continue;
		// scale to image size
		a.x *= SX; a.y *= SY;
		b.x *= SX; b.y *= SY;
		line(img, a, b, cv::Scalar(0, 200, 0), 2);
		circle(img, a, 3, cv::Scalar(0, 0, 200), -1);
		circle(img, b, 3, cv::Scalar(0, 0, 200), -1);
	}
}


// 返回格式化时间：20200426_150925
std::string OpenPose::getLocNameTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_name = cv::format("%d", t.tm_year + 1900) + cv::format("%.2d", t.tm_mon + 1) + cv::format("%.2d", t.tm_mday) + "_" +
		cv::format("%.2d", t.tm_hour) + cv::format("%.2d", t.tm_min) + cv::format("%.2d", t.tm_sec);
	return time_name;
}

void OpenPose::setViderWriterPara(const cv::Mat& img) {
	m_saveH = img.size().height;
	m_saveW = img.size().width;
	m_viderName = "./data/" + getLocNameTime() + ".avi";
	m_viderWriter = cv::VideoWriter(m_viderName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(m_saveW, m_saveH));
	m_frames = 0;
}

void OpenPose::saveVider(cv::Mat img, std::vector<cv::Point>& openPoseRet) {
	showDetectRet(img, openPoseRet);
	if ((m_saveH == 0) && (m_saveW == 0)) {
		setViderWriterPara(img);
		m_viderWriter << img;
	}
	else {
		if ((m_saveH != img.size().height) || (m_saveW != img.size().width)) {
			cv::resize(img, img, cv::Size(m_saveW, m_saveH));
			m_viderWriter << img;
		}
		else {
			m_viderWriter << img;
		}
	}

	++m_frames;
	if (m_frames == 25 * 60 * 10) {   // 每十分钟从新录制新视频
		m_saveH = 0;
		m_saveW = 0;
		m_viderWriter.release();
	}
}
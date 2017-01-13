
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "kdtree.h"

#define POINTS_IN_CLOUD (16*1024)
#define NUM_NEAREST_NEIGHBORS (4*1024)
#define UI_WIDTH  1024
#define UI_HEIGHT 600


kd::Point<2> MakePoint(double x, double y) {
    kd::Point<2> result;
    result[0] = x;
    result[1] = y;
    return result;
}

class UIKDTreeTest2D {
    std::vector<cv::Point2i> _pts;
    cv::Mat _curr_rgb;
    CvPoint _pCurr;
    cv::Mat _colorMap;
    kd::KDTree<2,size_t> _kd;
    static void mouseCallback(int event, int x, int y, int flags, void* param) {
        static_cast<UIKDTreeTest2D*>(param)->mouseEvent(event, x, y, flags,param);
    }
public:
    UIKDTreeTest2D() {
        cv::Mat img0 (1,NUM_NEAREST_NEIGHBORS,CV_8UC1);
        for (int i = 0 ; i < NUM_NEAREST_NEIGHBORS ; i++) {
            img0.data[i] = uchar(256*(i+1.f)/NUM_NEAREST_NEIGHBORS);
        }
        cv::applyColorMap(img0, _colorMap, cv::COLORMAP_RAINBOW);

        _curr_rgb = cv::Mat(UI_HEIGHT,UI_WIDTH,CV_8UC3);
        _pCurr = cv::Point(UI_WIDTH/2,UI_HEIGHT/2);
        cv::namedWindow("KDTree");
        cv::setMouseCallback("KDTree", &UIKDTreeTest2D::mouseCallback , this );
        std::vector<cv::Point2f> cvPointCloud;
        for (int i = 0; i < POINTS_IN_CLOUD; i++) {
            int x = rand() % UI_WIDTH;
            int y = rand() % UI_HEIGHT;
            _pts.push_back(cv::Point2f(float(x), float(y)));
        }

        for (int i = 0; i < POINTS_IN_CLOUD; i++) {
            _kd.insert(MakePoint(_pts[i].x,_pts[i].y),i);
        }
    }


    ~UIKDTreeTest2D() {
    }

    void mouseEvent( int event, int x, int y, int flags, void* ) {
        if(event == CV_EVENT_MOUSEMOVE ) {
            _pCurr = cvPoint(x,y);
        }
    }

    float dist2(const cv::Point2f &p1,const cv::Point2f &p2) {
        float x = p1.x - p2.x;
        float y = p1.y - p2.y;
        return x*x + y*y;
    }

    void run() {
        while(true) {
            _curr_rgb.setTo(0);
            try {
                std::vector<size_t> kd_i = _kd.kNNValue(MakePoint(_pCurr.x,_pCurr.y),NUM_NEAREST_NEIGHBORS);
                cv::line(_curr_rgb,cvPoint(_pCurr.x,0),cvPoint(_pCurr.x,_curr_rgb.rows),CV_RGB(255,0,255));
                cv::line(_curr_rgb,cvPoint(0,_pCurr.y),cvPoint(_curr_rgb.cols,_pCurr.y),CV_RGB(255,0,255));
                for(size_t i  = 0 ; i < _pts.size() ; i++ ) {
                    cv::circle(_curr_rgb,cv::Point(_pts[i].x,_pts[i].y),1,CV_RGB(128,128,128),1,CV_AA,0);
                }
                for (size_t i = 0 ; i < kd_i.size() ; i++) {
                    cv::circle(_curr_rgb,cv::Point(_pts[kd_i[i]].x,_pts[kd_i[i]].y),2,cv::Scalar(_colorMap.at<cv::Vec3b>(0,i)),1);
                }
            } catch (std::exception &e) {
                printf("Err : %s\n",e.what());
            }
            cv::imshow("KDTree",_curr_rgb);
            if( cv::waitKey(10) == 27 ) {
                return ;
            }
        }
    }
};

int main(int argc, char **argv) {
    UIKDTreeTest2D ui;
    ui.run();
    return EXIT_SUCCESS;
}

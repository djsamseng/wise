#include "desktop_capture.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>


DesktopCapture::DesktopCapture() {

}

DesktopCapture::~DesktopCapture() {
    cv::destroyAllWindows();
    std::cout << "DesktopCapture destroyed" << std::endl;
}

void DesktopCapture::getDesktopImage() {
    Display *display = XOpenDisplay(nullptr);
    Window window = DefaultRootWindow(display);
    XWindowAttributes attr = {0};
    XGetWindowAttributes(display, window, &attr);

    XImage *img = XGetImage(display, window,
        0, 0, attr.width, attr.height, AllPlanes, ZPixmap);
    std::cout << "Got img:" << img->data << std::endl;
    std::vector<std::uint8_t> pixels;
    pixels.resize(attr.width * attr.height * 4);
    memcpy(&pixels[0], img->data, pixels.size());

    cv::Mat mat = cv::Mat(attr.height, attr.width,
        (img->bits_per_pixel > 24 ? CV_8UC4 : CV_8UC3), &pixels[0]);

    // cv::namedWindow("Desktop capture", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display", mat);
    cv::waitKey(100);

    XDestroyImage(img);
    XCloseDisplay(display);
}
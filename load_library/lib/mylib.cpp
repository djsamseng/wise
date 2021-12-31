#include "mylib.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "audio_capture.h"
#include "desktop_capture.h"

extern "C" void entry_point_execute() {
    // Only destroyed once on program exit!
    static DesktopCapture desktopCapture;
    static AudioCapture audioCapture;

    std::cout << "Execute!" << std::endl;
    desktopCapture.getDesktopImage();
}

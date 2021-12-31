#ifndef __AUDIO_CAPTURE_H__
#define __AUDIO_CAPTURE_H__

#include <portaudio.h>

class AudioCapture {
    public:
        AudioCapture();
        ~AudioCapture();
    private:
        PaStream *d_stream;
};

#endif
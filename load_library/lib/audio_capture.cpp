#include "audio_capture.h"

#include <iostream>
#include <portaudio.h>

static int audioCallback(
    const void *inputBuffer,
    void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData
) {
    std::cout << "Got Audio data!" << std::endl;
    float *out = (float*)outputBuffer;
    float *in = (float*)inputBuffer;
    for (int i = 0; i < framesPerBuffer; i++) {
        out[i] = in[i];
    }
    return 0;
}

AudioCapture::AudioCapture() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    err = Pa_OpenDefaultStream(&this->d_stream,
        1, // Input channels
        1, // Output channels
        paFloat32,
        44100, // RATE
        1024, // frames per buffer
        audioCallback,
        nullptr);
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }

    err = Pa_StartStream(this->d_stream);
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
}

AudioCapture::~AudioCapture() {
    PaError err = Pa_StopStream(this->d_stream);
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    err = Pa_CloseStream(this->d_stream);
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    err = Pa_Terminate();
    if (err != paNoError) {
        std::cout << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
}
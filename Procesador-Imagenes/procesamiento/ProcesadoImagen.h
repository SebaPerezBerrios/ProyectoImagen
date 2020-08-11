#ifndef ProcesadoImagen_H
#define ProcesadoImagen_H

#include <opencv2/imgproc.hpp>

void escalarImagen(cv::Mat &imagen) { resize(imagen, imagen, cv::Size(), 1.33, 1.33); }

void escalaGrisesImagen(cv::Mat &imagen) { cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY); }

void difuminarImagen(cv::Mat &imagen, int intensidad) {
  GaussianBlur(imagen, imagen, cv::Size(intensidad, intensidad), 0, 0);
}

#endif
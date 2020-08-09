#ifndef Escalado_H
#define Escalado_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace escalado {

using namespace cv;

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);

  Mat imagenOriginal = imread(image_path, IMREAD_UNCHANGED);
  cv::cvtColor(imagenOriginal, imagenOriginal, cv::COLOR_RGB2RGBA);

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    // generar sub imagen a ser enviada, se incluyen los offsets
    auto regionEnviada = Rect(0, 0, imagenOriginal.cols, particiones[proceso]);
    Mat imagenEnviada = imagenOriginal(regionEnviada);

    // envio de imagenes a esclavos
    enviarImagenMPI(imagenEnviada, procesosReservados + proceso);

    // quitar particion previamente enviada de imagen original
    auto regionRestante =
        Rect(0, particiones[proceso], imagenOriginal.cols, imagenOriginal.rows - particiones[proceso]);
    imagenOriginal = imagenOriginal(regionRestante);
  }
}

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);

  resize(imagenRecibida, imagenRecibida, cv::Size(), 1.33, 1.33);

  enviarImagenMPI(imagenRecibida, 0);
}
}  // namespace escalado
#endif
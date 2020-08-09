#ifndef Difuminado_H
#define Difuminado_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace difuminado {

#define Blur 21

void enviarImagen(int procesosReservados, int procesosTotales, const std::string& nombreArchivo) {
  std::string image_path = samples::findFile(nombreArchivo);

  Mat imagenOriginal = imread(image_path, IMREAD_UNCHANGED);
  cv::cvtColor(imagenOriginal, imagenOriginal, cv::COLOR_RGB2RGBA);

  // offset usado para el blur, se envian regiones anexas para tener un difuminado suave entre particiones
  int offset = Blur / 2;

  int procesosEsclavos = procesosTotales - procesosReservados;

  // crear particionado para enviar a cada nodo, particionado por fila
  auto particiones = particionar(imagenOriginal.rows, procesosEsclavos);

  for (int proceso = 0; proceso < procesosEsclavos; proceso++) {
    // calculo de offsets segun particion
    int offsetAbajo = (proceso + 1 == procesosEsclavos) ? 0 : offset;
    int offsetArriba = (proceso == 0) ? 0 : offset;
    int offsetResto = (proceso == 0) ? offset : 0;

    // generar sub imagen a ser enviada, se incluyen los offsets
    auto regionEnviada = Rect(0, 0, imagenOriginal.cols, particiones[proceso] + offsetAbajo + offsetArriba);
    Mat imagenEnviada = imagenOriginal(regionEnviada);

    // envio de imagenes particionadas a esclavos junto a los datos de offset de cada particion
    enviarImagenMPI(imagenEnviada, procesosReservados + proceso);
    enviarIntMPI(offsetAbajo, procesosReservados + proceso);
    enviarIntMPI(offsetArriba, procesosReservados + proceso);

    // quitar particion previamente enviada de imagen original
    auto regionRestante = Rect(0, particiones[proceso] - offsetResto, imagenOriginal.cols,
                               imagenOriginal.rows - particiones[proceso] + offsetResto);
    imagenOriginal = imagenOriginal(regionRestante);
  }
}

void procesarImagen() {
  auto imagenRecibida = recibirImagenMPI(0);
  int offsetAbajo = recibirIntMPI(0);
  int offsetArriba = recibirIntMPI(0);

  // aplicar difuminado, quitar offsets y retornar al orquestador
  GaussianBlur(imagenRecibida, imagenRecibida, Size(Blur, Blur), 0, 0);

  auto quitarOffset = Rect(0, offsetArriba, imagenRecibida.cols, imagenRecibida.rows - offsetAbajo - offsetArriba);

  auto imagenRecortada = imagenRecibida(quitarOffset);

  enviarImagenMPI(imagenRecortada, 0);
}
}  // namespace difuminado
#endif
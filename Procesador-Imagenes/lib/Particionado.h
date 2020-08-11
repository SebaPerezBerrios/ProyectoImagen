#ifndef Particionado_H
#define Particionado_H

#include <vector>

std::vector<int> particionar(int cantidad, int numeroParticiones) {
  int particion = cantidad / numeroParticiones;
  std::vector<int> particiones(numeroParticiones, particion);

  auto resto = cantidad - particion * numeroParticiones;
  for (auto &particion : particiones) {
    if (resto--)
      particion++;
    else
      break;
  }
  return particiones;
}

std::vector<int> vectorAcumulador(const std::vector<int> particiones) {
  std::vector<int> acumulado(particiones.size());
  acumulado[0] = 0;

  for (size_t i = 1; i < particiones.size(); i++) {
    acumulado[i] = acumulado[i - 1] + particiones[i];
  }
  return acumulado;
}

#endif
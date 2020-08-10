# Proyecto de tratamiento de imágenes
## Descripción
Este repositorio contiene un programa que realiza las las operaciones de difuminado, escala de grises y escalado de imágenes, usando principalmente la librería `OpenCV` y distribuyendo el procesamiento usando una combinación de `OpenMPI` y `OpenMP`

## Requerimientos
El programa fue probado en Ubuntu 20.04. para su compilación se requiere la instalación de los siguientes paquetes.

```
sudo apt install libopencv-dev libopenmpi-dev libomp-dev
```

## Compilación

```
make -C  Procesador-Imagenes
```

## Ejecución
La carpeta del proyecto debe estar en una carpeta compartida visible por todas las máquinas definidas en el archivo `maquinas.txt`. Este archivo se debe editar según la configuración de red donde será ejecutado.

```
time ./run.sh
```
Editar el archivo run.sh para modificar imagen de entrada y operación a realizar. Alternativamente el programa se puede ejecutar de la siguiente forma.

```
mpirun --hostfile maquinas.txt ./Procesador-Imagenes/dist/programa <codigo operacion> <archivo imagen>
```

## Integrantes
- Léster Vasquez
- Ivan Pérez
- Sebastian Pérez

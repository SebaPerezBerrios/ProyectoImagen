#!/bin/bash

mpirun --hostfile maquinas.txt ./dist/programa test.png

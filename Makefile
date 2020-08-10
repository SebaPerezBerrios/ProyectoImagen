CXX = mpic++
CXXFLAGS = -O3 -fopenmp -Wall -I/usr/include/opencv4 -I/usr/include/opencv4/opencv
MKDIR = mkdir -p

LIBS = -lm -lmpi -lopencv_highgui -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core

directorios:
	$(MKDIR) build dist

main.o: directorios main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o build/main.o

all: clean main.o
	$(CXX) $(CXXFLAGS) -o dist/programa build/main.o $(LIBS)
	rm -fr build

clean:
	rm -fr *.o a.out core programa dist build

.DEFAULT_GOAL := all

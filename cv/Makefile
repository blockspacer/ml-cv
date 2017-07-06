
all:  joiner 

joiner: w1

w1: w1.o
	$(CXX) w1.o -o  $@ `PKG_CONFIG_PATH=/opt/opencv-2.4.2/lib/pkgconfig pkg-config opencv --libs`

w1.o: w1.cc
	$(CXX) -c -o w1.o w1.cc `PKG_CONFIG_PATH=/opt/opencv-2.4.2/lib/pkgconfig pkg-config opencv --cflags`

clean:
	rm -fr *.o w1

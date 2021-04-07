cd track
mkdir build
cd build
cmake ..
make
cp *.so ../../lib/
cd ../../

cd demo
mkdir build
cd build
cmake ..
make clean
make

# (c) 2016 Sergey Efilov (contaminated with 30% of v2k morons noise)
all:
	g++  train.cpp  -o train.exe -O3 -march=nocona   -g  -lfann -lm -Wfatal-errors -w -fpermissive
	gcc  create.c  -o create.exe  -O3 -march=nocona  -ggdb  -lfann -lm -Wfatal-errors -w
	g++ run.c -o run.exe -lfann -fpermissive -w
	#g++ find.cpp -o findnet.exe -lfann -fpermissive -w
	g++ data.cpp -o data.exe -lfann -fpermissive -w
	#g++ mutate.cpp -o mutate.exe -lfann -fpermissive -w
	#g++ work.c -o work.exe -lfann -fpermissive -w -lWinmm
#	gcc  fann_normal.c -O3 -march=nocona -lm -o fann_nor -g  -lfann
#	g++  cascade.c  -o cascade.exe -O3 -march=nocona   -g  -lfann -lm -Wfatal-errors -w
	#g++  -o lsnn.exe lsnn.c -O3 -march=nocona   -g  -lfann -lm -Wfatal-errors -w


cd ../code

python dct.py -f co -a paper_model -d cifar10 -b 128 -e 2 -nt symmetry -nr 0.5 -dm mmd -si 100000.0 -sr 0.25 -lto 1.0 -ltr 1.0

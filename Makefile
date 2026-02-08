FN ?= dbscan
BFN ?= benchmark

N ?= 100000
N_LIST ?= 1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000

C ?= 30
CS ?= 1.0
STD ?= 0.03
NR ?= 0.001
R ?= 0

init:
	cd script && \
	pip install -r requirements.txt

clean:
	@read -p "Delete folders data_in/ and data_out/? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -rf data_in/ data_out/; \
		echo "Folders deleted."; \
	else \
		echo "Operation cancelled."; \
	fi

generate:
	cd script && \
	python generator.py -n $(N) -c $(C) -cs $(CS) -std $(STD) -nr $(NR) -r $(R)

multiple-generate:
	cd script && \
	python generator.py -n $(N_LIST) -c $(C) -cs $(CS) -std $(STD) -nr $(NR) -r $(R)

plot:
	cd script && \
	python plot.py -fn $(FN)

plot-benchmark:
	cd script && \
	python benchmark.py -bfn $(BFN)

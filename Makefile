init:
	cd script && \
	python -m venv .venv && \
	call .venv/Scripts/activate && \
	pip install -r requirements.txt


FN_GEN ?= dataset
FN_PLOT ?= dbscan

N ?= 100000
N_LIST ?= 1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000

C ?= 30
CS ?= 10.0
STD ?= 0.3
NR ?= 0.001
R ?= 0

generate:
	cd script && \
	call .venv/Scripts/activate && \
	python generator.py -fn $(FN_GEN) -n $(N) -c $(C) -cs $(CS) -std $(STD) -nr $(NR) -r $(R)

multiple-generate:
	cd script && \
	call .venv/Scripts/activate && \
	python multiple_generator.py -fn $(FN_GEN) -n $(N_LIST) -c $(C) -cs $(CS) -std $(STD) -nr $(NR) -r $(R)

plot:
	cd script && \
	call .venv/Scripts/activate && \
	python plot.py -fn $(FN_PLOT)

generate-plot:
	make generate FN_GEN=$(FN_GEN) N=$(N) C=$(C) CS=$(CS) STD=$(STD) NR=$(NR) R=$(R) && \
	make plot FN_PLOT=$(FN_GEN)
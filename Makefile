init:
	cd script && \
	python -m venv .venv && \
	call .venv/Scripts/activate && \
	pip install -r requirements.txt

FN_IN ?= input
N ?= 100000
C ?= 5
generate:
	cd script && \
	call .venv/Scripts/activate && \
	python generate_points.py -fn $(FN_IN) -n $(N) -c $(C)

FN_OUT ?= output
plot:
	cd script && \
	call .venv/Scripts/activate && \
	python plot_clusters.py -fn $(FN_OUT)

gen-plot:
	make generate FN_IN=$(FN_IN) N=$(N) C=$(C) && \
	make plot FN_OUT=$(FN_IN)
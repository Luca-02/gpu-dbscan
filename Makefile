init:
	cd script && \
	python -m venv .venv && \
	call .venv/Scripts/activate && \
	pip install -r requirements.txt

FN_IN ?= input
NC ?= 20
XMIN ?= -5000
XMAX ?= 5000
YMIN ?= -5000
YMAX ?= 5000
generate:
	cd script && \
	call .venv/Scripts/activate && \
	python generate_points.py -fn $(FN_IN) -xmin $(XMIN) -xmax $(XMAX) -ymin $(YMIN) -ymax $(YMAX) -nc $(NC)

FN_OUT ?= output
plot:
	cd script && \
	call .venv/Scripts/activate && \
	python plot_clusters.py -fn $(FN_OUT)

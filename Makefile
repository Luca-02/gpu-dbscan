init:
	cd script && python -m venv .venv && call .venv/Scripts/activate && pip install -r requirements.txt

FN_INPUT ?= input
NC ?= 10
XMIN ?= -250
XMAX ?= 250
YMIN ?= -250
YMAX ?= 250
generate:
	cd script && call .venv/Scripts/activate && \
	python generate_points.py -fn $(FN_INPUT) -xmin $(XMIN) -xmax $(XMAX) -ymin $(YMIN) -ymax $(YMAX) -nc $(NC)

FN_OUTPUT ?= output
plot:
	cd script && call .venv/Scripts/activate && \
	python plot_clusters.py -fn $(FN_OUTPUT)

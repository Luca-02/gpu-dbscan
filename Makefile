init:
	cd script && python -m venv .venv && call .venv/Scripts/activate && pip install -r requirements.txt

FN_IN ?= input
NC ?= 30
XMIN ?= -250
XMAX ?= 250
YMIN ?= -250
YMAX ?= 250
generate:
	cd script && call .venv/Scripts/activate && \
	python generate_points.py -fn $(FN_IN) -xmin $(XMIN) -xmax $(XMAX) -ymin $(YMIN) -ymax $(YMAX) -nc $(NC)

FN_OUT ?= output
plot:
	cd script && call .venv/Scripts/activate && \
	python plot_clusters.py -fn $(FN_OUT)

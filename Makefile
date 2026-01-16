init:
	cd script && python -m venv .venv && call .venv/Scripts/activate && pip install -r requirements.txt

generate:
	cd script && call .venv/Scripts/activate && python generate_points.py

plot:
	cd script && call .venv/Scripts/activate && python plot_clusters.py

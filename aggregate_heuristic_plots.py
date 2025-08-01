import os
import re
import matplotlib.pyplot as plt
from PIL import Image

# Update this to your real plots root
PLOTS_DIR = r"C:\Users\Ryan\PycharmProjects\maisr\outputs\plots"
OUTPUT_DIR = os.path.join(PLOTS_DIR, "aggregated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_all_plots():
    all_plots = []
    for root, _, files in os.walk(PLOTS_DIR):
        for file in files:
            if file.lower().endswith(".png") and "heuristictest" in file.lower():
                all_plots.append(os.path.join(root, file))
    return all_plots

def extract_episode(filename):
    """Extract episode number from filename like heuristictest0_ep3.png"""
    m = re.search(r"_ep(\d+)", filename.lower())
    if m:
        return int(m.group(1))
    return None

def aggregate_by_level():
    plots = find_all_plots()
    level_dict = {}

    for path in plots:
        ep = extract_episode(path)
        if ep is None:
            continue
        # Make level zero-based (ep1 -> level0)
        level = ep - 1
        level_dict.setdefault(level, []).append(path)

    for level, files in level_dict.items():
        num_agents = len(files)
        cols = 6
        rows = (num_agents + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3))
        axes = axes.flatten()

        for idx, file in enumerate(sorted(files)):
            img = Image.open(file)
            axes[idx].imshow(img)
            agent_type = os.path.basename(os.path.dirname(os.path.dirname(file)))  # parent folder name
            axes[idx].set_title(agent_type, fontsize=7)
            axes[idx].axis('off')

        for j in range(num_agents, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, f"aggregated_level{level+1}.png")
        plt.savefig(output_file, dpi=200)
        plt.close(fig)
        print(f"Saved aggregated plot: {output_file}")

if __name__ == "__main__":
    aggregate_by_level()

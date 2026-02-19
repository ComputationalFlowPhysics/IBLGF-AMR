# import numpy as np
# import matplotlib.pyplot as plt
# import os


# workdir = os.getcwd()
# n_vec = {14: [4, 5], 30: [3, 4]}
# # n_vec = {14: [4,5]}
# gpu_np = {}
# gpu_np["A100"]=[2,4,5,6,8,10]
# gpu_np["A40"]=[2,4,5,6,8,10]
# gpu_np["CPU"]=[1,2,4,6]
# costperhour={"A100":66,"A40":33,"CPU":1}
# # gpu_folder={"A100":"bench", "A40":"bench-A40", "CPU":"bench-cpu"}
# gpu_folder={"A100":"build-A100", "A40":"build-A40", "CPU":"../longtalk012225/bench-cpu"}

# gpu_per_node=4
# cpu_per_node=96
# m={}
# time_filename="vortexRings/global_timings.txt"
# error_filename="vortexRings/global_error.txt"
# for gpu in gpu_np.keys():
#     m[gpu]={}
#     for n in n_vec:
#         m[gpu][n]={}
#         for block_power in n_vec[n]:
#             m[gpu][n][block_power]={}
#             m[gpu][n][block_power]["procs"]=[]
#             m[gpu][n][block_power]["time_1"]=[]
#             m[gpu][n][block_power]["time_2"]=[]
#             m[gpu][n][block_power]["error_L2"]=[]
#             m[gpu][n][block_power]["error_Linf"]=[]
#             m[gpu][n][block_power]["rate_1"]=[]
#             m[gpu][n][block_power]["rate_2"]=[]


# for gpu in gpu_np.keys():
#     gpu_foldername=gpu_folder[gpu]
#     gpu_npvec=gpu_np[gpu]
#     for n in n_vec:
#         for block_power in n_vec[n]:
#             case_folder=f"n{n}_bp{block_power}"
#             for nprocs in gpu_npvec:
#                 proc_folder=f"proc{nprocs*gpu_per_node+1}"
#                 if gpu=="CPU":
#                     nprocs_cpu=nprocs*cpu_per_node
#                     proc_folder=f"proc{nprocs_cpu}"
#                 full_folder=os.path.join(workdir,gpu_foldername,case_folder,proc_folder)
#                 #check it exists
#                 if not os.path.exists(full_folder):
#                     print(f"Folder does not exist: {full_folder}")
#                     m[gpu][n][block_power]["time_1"].append(np.nan)
#                     m[gpu][n][block_power]["time_2"].append(np.nan)
#                     m[gpu][n][block_power]["rate_1"].append(np.nan)
#                     m[gpu][n][block_power]["rate_2"].append(np.nan)
#                     m[gpu][n][block_power]["error_L2"].append(np.nan)
#                     m[gpu][n][block_power]["error_Linf"].append(np.nan)
#                     continue
#                 m[gpu][n][block_power]["procs"].append(nprocs*gpu_per_node+1)
#                 finished=os.path.exists(os.path.join(full_folder,time_filename)) and os.path.exists(os.path.join(full_folder,error_filename))
#                 if not finished:
#                     print(f"Case not finished: {full_folder}")
#                     m[gpu][n][block_power]["time_1"].append(np.nan)
#                     m[gpu][n][block_power]["time_2"].append(np.nan)
#                     m[gpu][n][block_power]["rate_1"].append(np.nan)
#                     m[gpu][n][block_power]["rate_2"].append(np.nan)
#                     m[gpu][n][block_power]["error_L2"].append(np.nan)
#                     m[gpu][n][block_power]["error_Linf"].append(np.nan)
#                     continue
#                 with open(os.path.join(full_folder,time_filename),"r") as f:
#                     line1=np.loadtxt(f,skiprows=1,max_rows=1)
#                     line2=np.loadtxt(f,skiprows=2,max_rows=1)
#                     # print(line1,line2)
#                     m[gpu][n][block_power]["time_1"].append(line1[1])
#                     m[gpu][n][block_power]["time_2"].append(line2[1])
#                     m[gpu][n][block_power]["rate_1"].append(line1[2])
#                     m[gpu][n][block_power]["rate_2"].append(line2[2])
#                 with open(os.path.join(full_folder,error_filename),"r") as f:
#                     line=np.loadtxt(f,skiprows=0,max_rows=1)
#                     m[gpu][n][block_power]["error_L2"].append(line[2])
#                     m[gpu][n][block_power]["error_Linf"].append(line[3])

# #make plots

                
                

                




# # --- Plot rate_1 vs ngpu/cores ---
# import matplotlib as mpl
# fig_rate1_ngpu, axs_rate1_ngpu = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
# n_list = sorted(n_vec.keys())
# gpu_colors = dict(zip(gpu_np.keys(), mpl.colormaps['tab10'].colors))
# bp_styles = ['-', '--', '-.', ':']
# lines_for_legend1 = []
# labels_for_legend1 = []
# for i, n in enumerate(n_list):
#     ax = axs_rate1_ngpu[i]
#     for gpu in m:
#         color = gpu_colors[gpu]
#         for ibp, bp in enumerate(m[gpu][n]):
#             data = m[gpu][n][bp]
#             if not data["procs"]:
#                 continue
#             ngpu = gpu_np[gpu]
#             style = bp_styles[ibp]
#             line, = ax.plot(ngpu, data["rate_1"], linestyle=style, marker='o', color=color,
#                             label=f"{gpu} (bp={bp})")
#             if i == 0:
#                 lines_for_legend1.append(line)
#                 labels_for_legend1.append(f"{gpu}, bp={bp}")
#     ax.set_xlabel("Number of GPUs/Cores", fontsize=12)
#     ax.set_title(f"n = {n}", fontsize=13)
#     ax.grid(True, which="both", ls="-", alpha=0.5)
# axs_rate1_ngpu[0].set_ylabel("Rate 1", fontsize=12)
# plt.tight_layout()
# filepath=os.path.abspath(__file__)
# fig_rate1_ngpu.subplots_adjust(right=0.8)
# fig_rate1_ngpu.legend(lines_for_legend1, labels_for_legend1, bbox_to_anchor=(0.8, 1), loc='upper left', fontsize='small', title="Type, Block Power")
# fig_rate1_ngpu.savefig("rate1_vs_ngpu.png", dpi=300)
# fig_rate1_ngpu.savefig(os.path.join(os.path.dirname(filepath), "rate1_vs_ngpu.png"), dpi=300)
# print("Plot successfully generated: rate1_vs_ngpu.png (split by n, unified legend)")

# # --- Plot rate_2 vs ngpu/cores ---
# fig_rate2_ngpu, axs_rate2_ngpu = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
# lines_for_legend2 = []
# labels_for_legend2 = []
# for i, n in enumerate(n_list):
#     ax = axs_rate2_ngpu[i]
#     for gpu in m:
#         color = gpu_colors[gpu]
#         for ibp, bp in enumerate(m[gpu][n]):
#             data = m[gpu][n][bp]
#             if not data["procs"]:
#                 continue
#             ngpu = gpu_np[gpu]
#             style = bp_styles[ibp]
#             line, = ax.plot(ngpu, data["rate_2"], linestyle=style, marker='o', color=color,
#                             label=f"{gpu} (bp={bp})")
#             if i == 0:
#                 lines_for_legend2.append(line)
#                 labels_for_legend2.append(f"{gpu}, bp={bp}")
#     ax.set_xlabel("Number of GPUs/Cores", fontsize=12)
#     ax.set_title(f"n = {n}", fontsize=13)
#     ax.grid(True, which="both", ls="-", alpha=0.5)
# axs_rate2_ngpu[0].set_ylabel("Rate 2", fontsize=12)
# plt.tight_layout()
# fig_rate2_ngpu.subplots_adjust(right=0.8)
# fig_rate2_ngpu.legend(lines_for_legend2, labels_for_legend2, bbox_to_anchor=(0.8, 1), loc='upper left', fontsize='small', title="Type, Block Power")
# fig_rate2_ngpu.savefig("rate2_vs_ngpu.png", dpi=300)
# fig_rate2_ngpu.savefig(os.path.join(os.path.dirname(filepath), "rate2_vs_ngpu.png"), dpi=300)
# print("Plot successfully generated: rate2_vs_ngpu.png (split by n, unified legend)")

# # cpus are 1 dollar per core hour, gpus are 66 dollars per gpu hour, plot rate vs cost




# # --- Plot rate_1 vs cost ---
# fig_rate1_cost, axs_rate1_cost = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
# lines_for_legend1_cost = []
# labels_for_legend1_cost = []
# for i, n in enumerate(n_list):
#     ax = axs_rate1_cost[i]
#     for gpu in m:
#         color = gpu_colors[gpu]
#         for ibp, bp in enumerate(m[gpu][n]):
#             data = m[gpu][n][bp]
#             if not data["procs"]:
#                 continue
#             ngpu = gpu_np[gpu]
#             style = bp_styles[ibp]
#             if gpu == "CPU":
#                 cost = [p * cpu_per_node for p in ngpu]
#             else:
#                 cost = [p * gpu_per_node * costperhour[gpu] for p in ngpu]
#             line, = ax.plot(cost, data["rate_1"], linestyle=style, marker='o', color=color,
#                             label=f"{gpu} (bp={bp})")
#             if i == 0:
#                 lines_for_legend1_cost.append(line)
#                 labels_for_legend1_cost.append(f"{gpu}, bp={bp}")
#     ax.set_xlabel("Credits per hour", fontsize=12)
#     ax.set_title(f"n = {n}", fontsize=13)
#     ax.grid(True, which="both", ls="-", alpha=0.5)
# axs_rate1_cost[0].set_ylabel("Rate 1", fontsize=12)
# plt.tight_layout()
# fig_rate1_cost.subplots_adjust(right=0.8)
# fig_rate1_cost.legend(lines_for_legend1_cost, labels_for_legend1_cost, bbox_to_anchor=(0.8, 1), loc='upper left', fontsize='small', title="Type, Block Power")
# fig_rate1_cost.savefig("rate1_vs_cost.png", dpi=300)
# fig_rate1_cost.savefig(os.path.join(os.path.dirname(filepath), "rate1_vs_cost.png"), dpi=300)
# print("Plot successfully generated: rate1_vs_cost.png (split by n, unified legend)")

# # --- Plot rate_2 vs cost ---
# fig_rate2_cost, axs_rate2_cost = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
# lines_for_legend2_cost = []
# labels_for_legend2_cost = []
# for i, n in enumerate(n_list):
#     ax = axs_rate2_cost[i]
#     for gpu in m:
#         color = gpu_colors[gpu]
#         for ibp, bp in enumerate(m[gpu][n]):
#             data = m[gpu][n][bp]
#             if not data["procs"]:
#                 continue
#             ngpu = gpu_np[gpu]
#             style = bp_styles[ibp]
#             if gpu == "CPU":
#                 cost = [p * cpu_per_node for p in ngpu]
#             else:
#                 cost = [p * gpu_per_node * costperhour[gpu] for p in ngpu]
#             line, = ax.plot(cost, data["rate_2"], linestyle=style, marker='o', color=color,
#                             label=f"{gpu} (bp={bp})")
#             if i == 0:
#                 lines_for_legend2_cost.append(line)
#                 labels_for_legend2_cost.append(f"{gpu}, bp={bp}")
#     ax.set_xlabel("Credits per hour", fontsize=12)
#     ax.set_title(f"n = {n}", fontsize=13)
#     ax.grid(True, which="both", ls="-", alpha=0.5)
# axs_rate2_cost[0].set_ylabel("Rate 2", fontsize=12)
# plt.tight_layout()
# fig_rate2_cost.subplots_adjust(right=0.8)
# fig_rate2_cost.legend(lines_for_legend2_cost, labels_for_legend2_cost, bbox_to_anchor=(0.8, 1), loc='upper left', fontsize='small', title="Type, Block Power")
# fig_rate2_cost.savefig("rate2_vs_cost.png", dpi=300)
# fig_rate2_cost.savefig(os.path.join(os.path.dirname(filepath), "rate2_vs_cost.png"), dpi=300)
# print("Plot successfully generated: rate2_vs_cost.png (split by n, unified legend)")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --- 1. Configuration & Setup ---
workdir = os.getcwd()
# Ensure plots save in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))

n_vec = {14: [4, 5], 30: [3, 4]}
gpu_np = {
    "A100": [2, 4, 5, 6, 8, 10],
    "A40": [2, 4, 5, 6, 8, 10],
    "CPU": [1, 2, 4, 6]
}
costperhour = {"A100": 66, "A40": 33, "CPU": 1}
gpu_folder = {
    "A100": "build-A100", 
    "A40": "build-A40", 
    "CPU": "../longtalk012225/bench-cpu"
}

gpu_per_node = 4
cpu_per_node = 96
time_filename = "vortexRings/global_timings.txt"
error_filename = "vortexRings/global_error.txt"

# Initialize Data Map
m = {gpu: {n: {bp: {
    "procs": [], "time_1": [], "time_2": [], 
    "error_L2": [], "error_Linf": [], 
    "rate_1": [], "rate_2": []
} for bp in n_vec[n]} for n in n_vec} for gpu in gpu_np}

# --- 2. Data Loading ---
for gpu, gpu_npvec in gpu_np.items():
    for n in n_vec:
        for block_power in n_vec[n]:
            case_folder = f"n{n}_bp{block_power}"
            for nprocs in gpu_npvec:
                # Folder naming logic
                if gpu == "CPU":
                    proc_folder = f"proc{nprocs * cpu_per_node}"
                else:
                    proc_folder = f"proc{nprocs * gpu_per_node + 1}"
                
                full_folder = os.path.join(workdir, gpu_folder[gpu], case_folder, proc_folder)
                target = m[gpu][n][block_power]

                time_path = os.path.join(full_folder, time_filename)
                error_path = os.path.join(full_folder, error_filename)

                if not (os.path.exists(time_path) and os.path.exists(error_path)):
                    continue

                try:
                    # Robust Timing Load: Read all lines, keep only those starting with numbers
                    with open(time_path, "r") as f:
                        raw_lines = [line.split() for line in f.readlines() if line.strip()]
                        # Filter lines where the first element is a digit (skips 'npts' headers)
                        data_lines = [l for l in raw_lines if l[0].replace('.','',1).isdigit()]
                        
                        if len(data_lines) >= 2:
                            target["procs"].append(nprocs)
                            target["time_1"].append(float(data_lines[0][1]))
                            target["rate_1"].append(float(data_lines[0][2]))
                            target["time_2"].append(float(data_lines[1][1]))
                            target["rate_2"].append(float(data_lines[1][2]))

                    # Error Load
                    with open(error_path, "r") as f:
                        line = np.loadtxt(f, skiprows=0, max_rows=1)
                        target["error_L2"].append(line[2])
                        target["error_Linf"].append(line[3])
                except Exception:
                    # Skip malformed folders silently to keep output clean
                    continue

# --- 3. Plotting Function ---
def plot_metric(m, metric_name, x_axis_type="ngpu", filename="plot.png"):
    # plt.style.use('ggplot') 
    n_list = sorted(n_vec.keys())
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    
    colors = dict(zip(gpu_np.keys(), mpl.colormaps['tab10'].colors))
    styles = ['-', '--', '-.', ':']
    legend_map = {}
    
    xlabel = "Cost (Credits per hour)" if x_axis_type == "cost" else "Number of GPUs / CPU Nodes"

    for i, n in enumerate(n_list):
        ax = axs[i]
        for gpu in m:
            for ibp, (bp, data) in enumerate(m[gpu][n].items()):
                if not data["procs"] or not data[metric_name]:
                    continue
                
                # Convert to numpy arrays for plotting
                x_raw = np.array(data["procs"])
                y_raw = np.array(data[metric_name])

                if x_axis_type == "cost":
                    x_vals = [p * (cpu_per_node if gpu == "CPU" else gpu_per_node * costperhour[gpu]) for p in x_raw]
                else:
                    x_vals = x_raw

                label_str = f"{gpu}, bp={bp}"
                line, = ax.plot(x_vals, y_raw, label=label_str, color=colors[gpu], 
                               linestyle=styles[ibp], marker='o', markersize=6)
                
                if i == 0 and label_str not in legend_map:
                    legend_map[label_str] = line

        ax.set_title(f"Grid Size $n = {n}$", fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        # if x_axis_type == "cost":
        #     ax.set_xscale('log')

    axs[0].set_ylabel(metric_name.replace("_", " ").title(), fontsize=12, fontweight='bold')
    
    # Create unified legend
    if legend_map:
        sorted_keys = sorted(legend_map.keys())
        fig.legend([legend_map[k] for k in sorted_keys], sorted_keys, 
                   loc='center left', bbox_to_anchor=(0.88, 0.5), title="Hardware, BP")
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Absolute path save fix
    save_path = os.path.join(script_dir, filename)
    fig.savefig(save_path, dpi=300)
    print(f"Successfully saved: {save_path}")
    plt.close(fig)

# --- 4. Execution ---
if __name__ == "__main__":
    plot_metric(m, "rate_1", "ngpu", "rate1_vs_ngpu.png")
    plot_metric(m, "rate_2", "ngpu", "rate2_vs_ngpu.png")
    plot_metric(m, "rate_1", "cost", "rate1_vs_cost.png")
    plot_metric(m, "rate_2", "cost", "rate2_vs_cost.png")
    plot_metric(m, "error_L2", "ngpu", "error_L2_vs_ngpu.png")
    plot_metric(m, "error_Linf", "ngpu", "error_Linf_vs_ngpu.png")
    print("\nProcessing complete. Check directory for PNG files.")
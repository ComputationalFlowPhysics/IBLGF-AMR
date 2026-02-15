import numpy as np
import matplotlib.pyplot as plt
import os


workdir = os.getcwd()
# n_vec = {14: [4, 5], 30: [3, 4]}
n_vec = {14: [4,5]}
gpu_np = {}
gpu_np["A100"]=[1,2,3,4,5]
gpu_np["A40"]=[1,2,3,4,5,6,10]
gpu_np["CPU"]=[1,2,4,6]
gpu_folder={"A100":"bench", "A40":"bench-A40", "CPU":"bench-cpu"}
gpu_per_node=4
cpu_per_node=96
m={}
time_filename="vortexRings/global_timings.txt"
error_filename="vortexRings/global_error.txt"
for gpu in gpu_np.keys():
    m[gpu]={}
    for n in n_vec:
        m[gpu][n]={}
        for block_power in n_vec[n]:
            m[gpu][n][block_power]={}
            m[gpu][n][block_power]["procs"]=[]
            m[gpu][n][block_power]["time_1"]=[]
            m[gpu][n][block_power]["time_2"]=[]
            m[gpu][n][block_power]["error_L2"]=[]
            m[gpu][n][block_power]["error_Linf"]=[]
            m[gpu][n][block_power]["rate_1"]=[]
            m[gpu][n][block_power]["rate_2"]=[]


for gpu in gpu_np.keys():
    gpu_foldername=gpu_folder[gpu]
    gpu_npvec=gpu_np[gpu]
    for n in n_vec:
        for block_power in n_vec[n]:
            case_folder=f"n{n}_bp{block_power}"
            for nprocs in gpu_npvec:
                proc_folder=f"proc{nprocs*gpu_per_node+1}"
                if gpu=="CPU":
                    nprocs_cpu=nprocs*cpu_per_node
                    proc_folder=f"proc{nprocs_cpu}"
                full_folder=os.path.join(workdir,gpu_foldername,case_folder,proc_folder)
                #check it exists
                if not os.path.exists(full_folder):
                    print(f"Folder does not exist: {full_folder}")
                    m[gpu][n][block_power]["time_1"].append(np.nan)
                    m[gpu][n][block_power]["time_2"].append(np.nan)
                    m[gpu][n][block_power]["rate_1"].append(np.nan)
                    m[gpu][n][block_power]["rate_2"].append(np.nan)
                    m[gpu][n][block_power]["error_L2"].append(np.nan)
                    m[gpu][n][block_power]["error_Linf"].append(np.nan)
                    continue
                m[gpu][n][block_power]["procs"].append(nprocs*gpu_per_node+1)
                finished=os.path.exists(os.path.join(full_folder,time_filename)) and os.path.exists(os.path.join(full_folder,error_filename))
                if not finished:
                    print(f"Case not finished: {full_folder}")
                    m[gpu][n][block_power]["time_1"].append(np.nan)
                    m[gpu][n][block_power]["time_2"].append(np.nan)
                    m[gpu][n][block_power]["rate_1"].append(np.nan)
                    m[gpu][n][block_power]["rate_2"].append(np.nan)
                    m[gpu][n][block_power]["error_L2"].append(np.nan)
                    m[gpu][n][block_power]["error_Linf"].append(np.nan)
                    continue
                with open(os.path.join(full_folder,time_filename),"r") as f:
                    line1=np.loadtxt(f,skiprows=1,max_rows=1)
                    line2=np.loadtxt(f,skiprows=2,max_rows=1)
                    # print(line1,line2)
                    m[gpu][n][block_power]["time_1"].append(line1[1])
                    m[gpu][n][block_power]["time_2"].append(line2[1])
                    m[gpu][n][block_power]["rate_1"].append(line1[2])
                    m[gpu][n][block_power]["rate_2"].append(line2[2])
                with open(os.path.join(full_folder,error_filename),"r") as f:
                    line=np.loadtxt(f,skiprows=0,max_rows=1)
                    m[gpu][n][block_power]["error_L2"].append(line[2])
                    m[gpu][n][block_power]["error_Linf"].append(line[3])

#make plots

                
                

                

plt.figure(figsize=(12, 7))

# Iterate through the nested dictionary to plot each case
for gpu in m:
    for n in m[gpu]:
        for bp in m[gpu][n]:
            data = m[gpu][n][bp]
            if not data["procs"]:
                continue
            
            # Use ngpu = nprocs * gpu_per_node for the x-axis
            # In your script, 'procs' stores nprocs*gpu_per_node + 1
            # We subtract 1 to get the actual GPU count
            # ngpu = [p - 1 for p in data["procs"]]
            ngpu=gpu_np[gpu]
            
            label_base = f"{gpu} (n={n}, bp={bp})"
            
            # Plot Rate 1
            # plt.plot(ngpu, data["rate_1"], 'o-', label=f"{label_base} - Rate 1")
            # Plot Rate 2
            plt.plot(ngpu, data["rate_2"], 's--', label=f"{label_base} - Rate 2")

plt.xlabel("Number of GPUs ($n_{gpu}$)", fontsize=12)
plt.ylabel("Rate", fontsize=12)
plt.title("Computational Rate vs. Number of GPUs", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()

# Save the plot
plt.savefig("rate_vs_ngpu.png", dpi=300)
print("Plot successfully generated: rate_vs_ngpu.png")
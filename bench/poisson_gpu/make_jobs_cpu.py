# # import os

# # n_proc = {}
# # # n_vec[nodes] = [block_powers]
# # # n_vec = {14: [4, 5], 30: [3, 4]}
# # n_vec = {14: [ 4,5], 30: [3, 4]}

# # # Assign the same parameter space to different process counts
# # for p in [2, 4]:
# #     n_proc[p] = n_vec
# # bench_dir = os.getcwd()
# # for nprocs in n_proc:
# #     job_filename = f"job_nprocs{nprocs}.sh"
# #     job_folder = "./"
    
# #     with open(os.path.join(job_folder, job_filename), "w") as job_file:
# #         # 1. Slurm Header
# #         job_file.write("#!/bin/bash\n")
# #         job_file.write(f"export APPTAINERENV_LD_LIBRARY_PATH=\"/usr/local/lib:$LD_LIBRARY_PATH\"\n")
# #         # job_file.write(f"#SBATCH --job-name=proc{nprocs}\n")
# #         # job_file.write(f"#SBATCH --ntasks={nprocs}\n")
# #         # job_file.write("#SBATCH --time=01:00:00\n")
# #         # job_file.write("#SBATCH --output=log_%j.out\n\n")


# #         # 2. Command Generation
# #         for n in n_proc[nprocs]:
# #             for block_power in n_proc[nprocs][n]:
# #                 # Create the directory structure
# #                 parent_folder = f"n{n}_bp{block_power}"
# #                 subfolder = f"proc{nprocs}"
# #                 foldername = os.path.join(parent_folder, subfolder)
# #                 os.makedirs(foldername, exist_ok=True)
                
# #                 # Write the execution line
# #                 # We use 'cd' to ensure output files stay in the specific folder
# #                 job_file.write(f"echo 'Running n={n}, bp={block_power} on {nprocs} procs'\n")
# #                 job_file.write(f"cd {os.path.abspath(foldername)}\n")
# #                 job_file.write(f"srun --ntasks={nprocs} --mpi=pmix  --export=ALL,OMPI_MCA_psec=^munge,PMIX_MCA_psec=^munge,OMPI_MCA_btl_vader_single_copy_mechanism=none apptainer exec --nv --bind {bench_dir} ../../my-app_gpu.sif ../../poisson_gpu.x ../config.cf\n")
# #                 job_file.write("cd ../..\n")
# import os

# # n_vec[n_size] = [block_powers]
# n_vec = {14: [4, 5], 30: [3, 4]}

# # CONFIGURATION
# nodes_to_test = [2, 4]   # Total physical nodes
# gpus_per_node = 2        # Delta standard (match your salloc request)

# bench_dir = os.getcwd()

# for nnodes in nodes_to_test:
#     # Logic: One node has (gpus_per_node + 1) tasks
#     # The other (nnodes - 1) nodes have (gpus_per_node) tasks
#     nprocs = (gpus_per_node + 1) + ((nnodes - 1) * gpus_per_node)
    
#     job_filename = f"job_nodes{nnodes}_tasks{nprocs}.sh"
    
#     with open(job_filename, "w") as job_file:
#         job_file.write("#!/bin/bash\n")
#         job_file.write(f"export APPTAINERENV_LD_LIBRARY_PATH=\"/usr/local/lib:$LD_LIBRARY_PATH\"\n")

#         for n in n_vec:
#             for block_power in n_vec[n]:
#                 parent_folder = f"n{n}_bp{block_power}"
#                 subfolder = f"proc{nprocs}"
#                 foldername = os.path.join(parent_folder, subfolder)
#                 os.makedirs(foldername, exist_ok=True)
                
#                 job_file.write(f"echo 'Running n={n}, bp={block_power} on {nprocs} tasks across {nnodes} nodes'\n")
#                 job_file.write(f"cd {os.path.abspath(foldername)}\n")

#                 # --- HETEROGENEOUS SRUN LOGIC ---
#                 export_flags = "--export=ALL,OMPI_MCA_psec=^munge,PMIX_MCA_psec=^munge,OMPI_MCA_btl_vader_single_copy_mechanism=none"
#                 app_cmd = f"apptainer exec --nv --bind {bench_dir} ../../my-app_gpu.sif ../../poisson_gpu.x ../config.cf"
                
#                 tasks_special = gpus_per_node + 1
#                 tasks_regular = gpus_per_node
#                 num_regular_nodes = nnodes - 1
                
#                 if num_regular_nodes > 0:
#                     # Component 0: 1 node, G+1 tasks
#                     # Component 1: Remainder nodes, G tasks each
#                     srun_line = (
#                         f"srun {export_flags} --nodes=1 --ntasks={tasks_special} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd} : "
#                         f"--nodes={num_regular_nodes} --ntasks-per-node={tasks_regular} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd}\n"
#                     )
#                 else:
#                     # Single node case: Just the G+1 tasks
#                     srun_line = f"srun {export_flags} --nodes=1 --ntasks={tasks_special} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd}\n"
                
#                 job_file.write(srun_line)
#                 job_file.write("cd ../..\n")

import os

# n_vec[n_size] = [block_powers]
n_vec = {14: [4, 5,6], 30: [3, 4,5]}

# CONFIGURATION
nodes_to_test = [2,4,6,10]   # Total physical nodes
cpus_per_node = 96        # Delta standard 
bench_dir = os.getcwd()

# Define absolute paths to ensure the srun finds them from subfolders
app_path = os.path.abspath("my-app_cpu.sif")
exe_path = os.path.abspath("poisson.x")

for nnodes in nodes_to_test:
    # Logic: One node has (gpus_per_node + 1) tasks, others have (gpus_per_node)
    nprocs = cpus_per_node*nnodes
    
    # 1. GENERATE THE EXECUTION SCRIPT (.sh)
    job_filename = f"job_nodes{nnodes}_tasks{nprocs}.sh"
    with open(job_filename, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write(f"export APPTAINERENV_LD_LIBRARY_PATH=\"/usr/local/lib:$LD_LIBRARY_PATH\"\n")

        for n in n_vec:
            for block_power in n_vec[n]:
                parent_folder = f"n{n}_bp{block_power}"
                subfolder = f"proc{nprocs}"
                foldername = os.path.join(parent_folder, subfolder)
                os.makedirs(foldername, exist_ok=True)
                
                job_file.write(f"echo 'Running n={n}, bp={block_power} on {nprocs} tasks'\n")
                job_file.write(f"cd {os.path.abspath(foldername)}\n")

                export_flags = "--export=ALL,OMPI_MCA_psec=^munge,PMIX_MCA_psec=^munge,OMPI_MCA_btl_vader_single_copy_mechanism=none"
                # Updated to use absolute paths for the .sif and .x
                app_cmd = f"apptainer exec --bind {bench_dir} {app_path} {exe_path} ../config.cf"
                
                srun_line = f"srun {export_flags} --nodes={nnodes} --ntasks={nprocs} --cpus-per-task=1 --mpi=pmix {app_cmd}\n"
                # tasks_special = gpus_per_node + 1
                # tasks_regular = gpus_per_node
                # num_regular_nodes = nnodes - 1
                
                # if num_regular_nodes > 0:
                #     srun_line = (
                #         f"srun {export_flags} --nodes=1 --ntasks={tasks_special} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd} : "
                #         f"--nodes={num_regular_nodes} --ntasks-per-node={tasks_regular} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd}\n"
                #     )
                # else:
                #     srun_line = f"srun {export_flags} --nodes=1 --ntasks={tasks_special} --gpus-per-node={gpus_per_node} --mpi=pmix {app_cmd}\n"
                
                job_file.write(srun_line)
                job_file.write("rm -f vortexRings/flow_mesh*\n")
                job_file.write("cd ../..\n")

    # 2. GENERATE THE SBATCH WRAPPER (.sbatch)
    sbatch_filename = f"submit_nodes{nnodes}.sbatch"
    with open(sbatch_filename, "w") as sbatch_file:
        sbatch_file.write("#!/bin/bash\n")
        sbatch_file.write(f"#SBATCH --job-name=poisson_n{nnodes}\n")
        sbatch_file.write("#SBATCH --account=bfzi-delta-cpu\n")
        sbatch_file.write("#SBATCH --partition=cpu\n")
        sbatch_file.write(f"#SBATCH --nodes={nnodes}\n")
        sbatch_file.write(f"#SBATCH --tasks-per-node={cpus_per_node}\n")
        sbatch_file.write("#SBATCH --cpus-per-task=1\n")
        sbatch_file.write("#SBATCH --mem=0\n")
        sbatch_file.write("#SBATCH --time=04:00:00\n")
        sbatch_file.write(f"#SBATCH --output=run_n{nnodes}_%j.log\n\n")
        
        sbatch_file.write(f"chmod +x {job_filename}\n")
        sbatch_file.write(f"./{job_filename}\n")

    print(f"Created {job_filename} and {sbatch_filename}")
    # submit it
    os.system(f"sbatch {sbatch_filename}")
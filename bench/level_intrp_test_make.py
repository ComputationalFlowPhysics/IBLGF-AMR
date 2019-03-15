#!/usr/bin/env python
N = [8]
refine_level = [1,2,3,4,5,6,7,8,9]

f_bash = open("run.q", "w")
f_bash.write("#!/bin/bash \n")
f_bash.write("mkdir -p ./refine_corretion_grid_convergence \n")

#for N_blocks in [4, 8, 16, 32]:
for N_blocks in [8]:
    for n_refine in refine_level:
        for n in N:
        #for N_blocks in [20, 32]:
            s = "cf_{}_{}_refine_{}".format(n, N_blocks, n_refine)
            f = open(s, "w")
            f.write("""simulation_parameters
{{
    nLevels={n_refine};
    L=4;
    domain{{

        Lx=8;

        max_extent   = 1024;

        block_extent = {ni};

        block
        {{
            base   = (0,0,0);
            extent = ({b},{b},{b});
        }}
    }}
}}
        """.format(n_refine = n_refine, ni=n, b = N_blocks*n ) )

            f_bash.write("../bin/iblgf.x {ss} > ./refine_corretion_grid_convergence/{ss}.out \n".format(ss=s))

            f.close()

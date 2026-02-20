import os

# n_vec={14:[4,5],30:[3,4]}
n_vec={14:[4,5,6],30:[3,4,5]}
for n in n_vec:
    for block_power in n_vec[n]:
        print(n,block_power,n*(2**block_power),(n*(2**block_power))**3)
        n_side=n*(2**block_power)
        bd_base=( -0.5*n_side, -0.5*n_side, -0.5*n_side )
        bd_extent=( n_side, n_side, n_side )
        block_base=( -0.5*n_side, -0.5*n_side, -0.5*n_side )
        block_extent=( n_side, n_side, n_side )
        config_content = f"""simulation_parameters
        {{
            nLevels=0;
            global_refinement=0;
            refinement_factor=0.125;
            #correction=true;
            #subtract_non_leaf=true;

            output
            {{
                directory=vortexRings;
            }}

            vortex
            {{
                c1=1000000.0;
                c2=15;
                R=0.125;
                center=(0,0,0);
            }}

            domain
            {{
                Lx=1.0;
                bd_base=({bd_base[0]},{bd_base[1]},{bd_base[2]});
                bd_extent=({bd_extent[0]},{bd_extent[1]},{bd_extent[2]});

                block_extent={n};

                block
                {{
                    base=({block_base[0]},{block_base[1]},{block_base[2]});
                    extent=({block_extent[0]},{block_extent[1]},{block_extent[2]});
                }}
            }}

            EXP_LInf=0.0116450;
        }}"""
        #make directory and write file
        foldername=f"n{n}_bp{block_power}"
        os.makedirs(foldername, exist_ok=True)
        with open(os.path.join(foldername,"config.cf"),"w") as f:
            f.write(config_content)
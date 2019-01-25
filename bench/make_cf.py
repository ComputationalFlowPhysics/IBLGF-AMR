N = [11, 15, 21, 26]
#N = [11, 15]
#N = [6]

f_bash = open("run.q", "w")

for block in [4, 8, 16]:
    for n in N:
    #for block in [20, 32]:
        s = "cf_{}_{}".format(n, block)
        f = open(s, "w")
        f.write("""simulation_parameters
    {{
        nLevels=0;
        L=4;
        domain{{

            Lx=3;

            max_extent   = 1280;

            block_extent = {ni};

            block
            {{
                base   = (0,0,0);
                extent = ({b},{b},{b});
            }}
        }}
    }}
    """.format(ni=n, b = block*n ) )

        f_bash.write("../bin/iblgf.x {ss} >> ./fourier_cont_Nb_0.8/{ss}.out \n".format(ss=s))

        f.close()

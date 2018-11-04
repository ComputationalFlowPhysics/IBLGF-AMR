N = [11, 15, 21, 26]
#N = [11, 15]
#N = [11]

f_bash = open("run.q", "w")

for n in N:
    for block in [2, 4, 8, 16, 32]:
    #for block in [20, 32]:
        s = "cf_{}_{}".format(n, block)
        f = open(s, "w")
        f.write("""simulation_parameters
    {{
        nLevels=0;
        L=5;
        domain{{

            Lx=5;

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

        f_bash.write("../bin/iblgf.x {ss} >> ./sin_results/{ss}.out \n".format(ss=s))

        f.close()

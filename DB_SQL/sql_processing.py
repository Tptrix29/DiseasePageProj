import pandas as pd
import numpy as np

AD = pd.read_csv("SourceData/oasis_longitudinal.csv", header=0)
sql_script = "AD.sql"
r, c = AD.shape
AD = AD.fillna("NULL")
f = open(sql_script, 'w')
for i in range(r):
    vals = [str(i) for i in AD.iloc[i, :]]
    f.write("INSERT INTO ADPred(" +
            "SBJ_ID, MRI_ID, visit, MR_Delay, sex, handUsage, age, eduYear, scioEco, MMSE, CDR, eTIV, nWBV, ASF, disorderRank" +
            ") \nVALUES('" +
            vals[0] + "', '" +
            vals[1] + "', " +
            vals[3] + ", " +
            vals[4] + ", '" + vals[5] + "', '" + vals[6] + "', " +
            ", ".join(vals[7:]) + ", '" + vals[2]
            + "');\n")
f.close()

PD = pd.read_csv("SourceData/parkinsons.csv", header=0)
script_name = "PD.sql"
PD = PD.fillna("NULL")
r, c = PD.shape
f = open(script_name, 'w')
for i in range(r):
    vals = [str(i) for i in PD.iloc[i, :]]
    f.write("INSERT INTO PDAudioPred\nVALUES('" +
            vals[0] + "', " + ", ".join(vals[1:]) +
            ");\n")

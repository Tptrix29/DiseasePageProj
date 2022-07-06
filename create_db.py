from config import sql_script, sql_script_dir

print("Generating SQL Command for you......\n")
for sql in sql_script:
    current_path = '/'.join(str(__file__).split("/")[:-1])
    cmd = "source " + current_path + sql_script_dir + sql + ";"
    print(cmd)

print("\nSuccessfully Done!")


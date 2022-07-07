# MySQL DataBase Configuration
DB_CONFIG = {
    'DB_USER': 'root',
    'DB_IP':  'localhost',
    'DB_PORT': 3306,
    'DB_PASSWD': 'TpZl7530'
}

# Script files
sql_script_dir = "/DB_SQL/"
sql_script = [i + ".sql" for i in ["Init", "AD", "PD"]]

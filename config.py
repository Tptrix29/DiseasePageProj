# MySQL DataBase Configuration
DB_CONFIG = {
    'DB_IP':  'localhost',   # MySQL服务器IP，默认本地IP
    'DB_PORT': 3306,   # MySQL服务器端口，默认3306
    'DB_USER': 'root',   # MySQL用户名，默认root
    'DB_PASSWD': 'TpZl7530'  # MySQL用户密码，请输入您设置的密码，必须修改！！！
}

# Script files
sql_script_dir = "/DB_SQL/"
sql_script = [i + ".sql" for i in ["Init", "AD", "PD"]]

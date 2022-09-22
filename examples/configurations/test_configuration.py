import configparser

# CREATE OBJECT
config_file = configparser.ConfigParser()

# READ CONFIG FILE
config_file.read("hidpy.cfg")

print(config_file)